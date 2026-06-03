#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
if os.name == "nt":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import datetime
import io as std_io
import json
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transf
from pytorch_model_summary import summary
from torch.utils.data import DataLoader
from tqdm import tqdm

from DataFeed import DataFeed, pool_beam_soft_for_downsample
from MyFunc import *
from model_icassp import *
from model_pp import *
from model_ResNet import *


def parse_args():
    parser = argparse.ArgumentParser(description="Image-only training")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--train_batch_size", type=int, default=6)
    parser.add_argument("--test_batch_size", type=int, default=6)
    parser.add_argument("--lr", type=float, default=7.5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--loss_type", type=str, default="focal", choices=["crossentropy", "focal"])
    parser.add_argument("--grad_clip", type=float, default=10.0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--use_early_stopping", action="store_true", default=True)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--augment_data", action="store_true", default=True)
    parser.add_argument("--label_smoothing", action="store_true", default=True)
    parser.add_argument("--soft_label_weight", type=float, default=0.2)
    parser.add_argument("--beam_soft_label_temperature", type=float, default=2.0)

    # Model parameters
    parser.add_argument("--feature_size", type=int, default=64)
    parser.add_argument("--gru_hidden_size", type=int, default=64)
    parser.add_argument("--gru_num_layers", type=int, default=2)
    parser.add_argument("--num_classes", type=int, default=64)
    parser.add_argument("--seq_length", type=int, default=8)
    parser.add_argument("--num_pred", type=int, default=3)
    parser.add_argument("--downsample_ratio", type=int, default=1)
    parser.add_argument("--full_beam_dim", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--ds_conv_block_version", type=int, default=2)
    parser.add_argument(
        "--model_name",
        type=str,
        default="ImageModalityNet_MHA",
        choices=["DirectStem16Image_v2", "ImageModalityNet_MHA", "ImageOnlyAblationFusionNet"],
    )

    # Dataset parameters
    parser.add_argument("--dataset_pct", type=float, default=1.0)
    parser.add_argument("--train_csv_name", type=str, default="train_seqs_RA.csv")
    parser.add_argument("--test_csv_name", type=str, default="test_seqs_RA.csv")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--use_gpu", action="store_true", default=True)
    parser.add_argument("--save_dir", type=str, default="saved_folder_train")
    parser.add_argument("--debug", action="store_true", default=True)
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--start_epoch", type=int, default=0)

    # Learning rate scheduler
    parser.add_argument("--T_0", type=int, default=10)
    parser.add_argument("--T_mult", type=int, default=2)
    parser.add_argument("--eta_min", type=float, default=1e-6)
    return parser.parse_args()


def _morph_dilate(x, k=3):
    if x.dim() >= 3 and x.shape[-3] != 1:
        x = x.unsqueeze(-3)
    return F.max_pool2d(x, kernel_size=k, stride=1, padding=k // 2)


def _morph_erode(x, k=3):
    if x.dim() >= 3 and x.shape[-3] != 1:
        x = x.unsqueeze(-3)
    return -F.max_pool2d(-x, kernel_size=k, stride=1, padding=k // 2)


def _gaussian_blur_2d(x, sigma):
    if sigma <= 0:
        return x
    radius = max(1, int(3 * sigma))
    size = 2 * radius + 1
    t = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    g = torch.exp(-(t ** 2) / (2 * (sigma ** 2)))
    g = g / g.sum()
    if x.dim() >= 3 and x.shape[-3] != 1:
        x = x.unsqueeze(-3)
    c = x.shape[-3]
    kx = g.view(1, 1, 1, size).expand(c, 1, 1, size)
    ky = g.view(1, 1, size, 1).expand(c, 1, size, 1)
    y = F.conv2d(x, kx, padding=(0, radius), groups=c)
    return F.conv2d(y, ky, padding=(radius, 0), groups=c)


def augment_image_sequence_in_loop(img):
    if img.dim() != 5:
        return img
    b, t, _, _, _ = img.shape
    out = img
    do_morph = torch.rand(b, device=img.device) < 0.6
    if do_morph.any():
        op = torch.randint(0, 3, (b,), device=img.device)
        for bi in torch.nonzero(do_morph, as_tuple=False).flatten().tolist():
            if op[bi].item() == 1:
                out[bi] = _morph_erode(out[bi], k=3)
            elif op[bi].item() == 2:
                out[bi] = _morph_dilate(out[bi], k=3)
    do_blur = torch.rand(b, device=img.device) < 0.4
    if do_blur.any():
        sigmas = (0.5 + 1.5 * torch.rand(b, device=img.device)).tolist()
        for bi in torch.nonzero(do_blur, as_tuple=False).flatten().tolist():
            out[bi] = _gaussian_blur_2d(out[bi], float(sigmas[bi]))
    mask = (torch.rand(b, t, 1, 1, 1, device=img.device) < 0.5).to(img.dtype)
    return (out + mask * (torch.randn_like(out) * 0.01)).clamp(0.0, 1.0)


def align_pool_label_soft(label_soft, args):
    t_need = 1 + args.num_pred
    t = label_soft.shape[1]
    if t < t_need:
        rep = label_soft[:, -1:, :].expand(-1, t_need - t, -1).clone()
        label_soft = torch.cat([label_soft, rep], dim=1)
    elif t > t_need:
        label_soft = label_soft[:, :t_need]
    return pool_beam_soft_for_downsample(label_soft.float(), args.downsample_ratio, args.num_classes, full_dim=args.full_beam_dim)


def _hard_supervision_criterion(args):
    if args.loss_type == "focal":
        return FocalLoss(alpha=1, gamma=2, ignore_index=-100)
    return nn.CrossEntropyLoss(ignore_index=-100)


def _soft_ce(logits, target_soft):
    log_p = F.log_softmax(logits, dim=1)
    q = target_soft.to(dtype=log_p.dtype, device=log_p.device)
    return (-(q * log_p).sum(dim=1)).mean()


def _build_image_batch(image_masks, args, device, augment=False):
    img = image_masks.unsqueeze(2)
    if augment:
        img = augment_image_sequence_in_loop(img)
    b, _, c, h, w = img.shape
    return torch.cat([img, torch.zeros(b, args.num_pred, c, h, w, device=img.device)], dim=1).to(device)


def _prepare_eval_labels(beam, label, args, device):
    beam_down = torch.floor(beam.float() / args.downsample_ratio).to(torch.int64)
    label_down = torch.floor(label.float() / args.downsample_ratio).to(torch.int64)
    if args.num_pred > label_down.shape[1]:
        raise ValueError(
            f"num_pred={args.num_pred} exceeds label time dimension {label_down.shape[1]}"
        )
    return torch.cat([beam_down[..., -1:], label_down[:, : args.num_pred]], dim=-1).to(device)


def _aggregate_eval_metrics(model, dataloader, args, device, *, show_progress=False, desc="Eval"):
    """Image-only eval: same batching and loss as train_model (no augment, no soft label)."""
    model.eval()
    criterion = _hard_supervision_criterion(args)
    all_outputs, all_labels = [], []
    loss_sum = 0.0
    iterator = tqdm(dataloader, unit="batch", file=sys.stdout, desc=desc) if show_progress else dataloader

    for batch in iterator:
        image_masks, beam, label, _ = batch[:4]
        test_label = _prepare_eval_labels(beam, label, args, device)
        with torch.no_grad():
            outputs, _, _ = model(_build_image_batch(image_masks, args, device, augment=False), None)
        outputs = outputs[:, -(args.num_pred + 1) :, :]
        loss_sum += criterion(outputs.reshape(-1, args.num_classes), test_label.flatten()).item()
        all_outputs.append(outputs)
        all_labels.append(test_label)

    if len(dataloader) == 0:
        raise ValueError("dataloader is empty")

    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    avg_loss = loss_sum / len(dataloader)
    topk_acc, _ = calculate_topk_accuracy(all_outputs, all_labels)
    dba_score = calculate_dba_score(all_outputs, all_labels)
    return avg_loss, topk_acc, dba_score


def _print_eval_summary(loss_title, loss, topk_acc, dba_score):
    print(f"{loss_title}: {loss:.4f}", flush=True)
    print("DBA-Score (Top-3):", dba_score)
    print("Top-K Accuracy:", flush=True)
    for k, acc in topk_acc.items():
        print(f"Top-{k}: {acc}", flush=True)


def _append_results_txt(save_path, *, title, loss, topk_acc, dba_score, filename="training_results.txt"):
    """Append validation / test metrics to a text log under save_path."""
    results_path = os.path.join(save_path, filename)
    with open(results_path, "a", encoding="utf-8") as f:
        f.write(f"{title}\n\n")
        f.write(f"Test Loss: {loss:.4f}\n\n")
        dba_str = ", ".join([f"{x:.4f}" for x in dba_score])
        f.write(f"DBA-Score (Top-3): [{dba_str}]\n\n")
        f.write("Top-K Accuracy Per Time Slot:\n")
        for k, acc in topk_acc.items():
            acc_str = ", ".join([f"{a:.4f}" for a in acc])
            f.write(f"Top-{k} Accuracy: [{acc_str}]\n")
        f.write("=" * 50 + "\n\n")


def train_model(student_model, dataloaders, args, optimizer, scheduler, device, save_path):
    criterion = _hard_supervision_criterion(args)
    start_epoch, best_test_loss = args.start_epoch, 1e3
    if args.resume:
        start_epoch, best_test_loss = load_checkpoint(save_path, student_model, optimizer, scheduler)
    train_acc_all, train_loss_all, train_task_loss_all = [], [], []
    val_acc_all, val_loss_all, lrs = [], [], []
    epochs_without_improvement = 0

    results_txt = os.path.join(save_path, "training_results.txt")
    with open(results_txt, "w", encoding="utf-8") as f:
        f.write("Image-only beam prediction — validation / test log\n")
        f.write(f"seq_len={args.seq_length}, num_pred={args.num_pred}, num_classes={args.num_classes}\n")
        f.write("=" * 50 + "\n\n")

    for epoch in range(start_epoch, args.epochs):
        student_model.train()
        running_loss, running_task_loss, running_acc = 0.0, 0.0, 0.0
        lrs.append(optimizer.param_groups[0]["lr"])
        with tqdm(dataloaders["train"], unit="batch", file=sys.stdout) as tepoch:
            for i, (image_masks, beam, label, label_soft) in enumerate(tepoch, 0):
                tepoch.set_description(f"Epoch {epoch}")
                beam_down = torch.floor(beam.float() / args.downsample_ratio).to(torch.int64)
                label_down = torch.floor(label.float() / args.downsample_ratio).to(torch.int64)
                train_label = torch.cat([beam_down[..., -1:], label_down[:, :args.num_pred]], dim=-1).to(device)

                image_batch = _build_image_batch(image_masks, args, device, augment=args.augment_data)
                target_soft = align_pool_label_soft(label_soft, args).to(device)

                optimizer.zero_grad()
                outputs, _, _ = student_model(image_batch, None)
                outputs = outputs[:, -(args.num_pred + 1):, :]
                logits = outputs.reshape(-1, args.num_classes)
                hard = criterion(logits, train_label.flatten())
                if args.label_smoothing:
                    soft = _soft_ce(logits, target_soft.reshape(-1, args.num_classes))
                    total_loss = (1.0 - args.soft_label_weight) * hard + args.soft_label_weight * soft
                else:
                    total_loss = hard
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=args.grad_clip)
                optimizer.step()

                topk_acc, _ = calculate_topk_accuracy(outputs, train_label)
                running_loss = (total_loss.item() + i * running_loss) / (i + 1)
                running_task_loss = (hard.item() + i * running_task_loss) / (i + 1)
                running_acc = (topk_acc[1].mean() + i * running_acc) / (i + 1)
                tepoch.set_postfix(loss=running_loss, hard=running_task_loss, top1=running_acc)

        train_loss_all.append(running_loss)
        train_task_loss_all.append(running_task_loss)
        train_acc_all.append(running_acc)
        scheduler.step()
        val_loss, val_topk, _ = validate_model(epoch, student_model, dataloaders["test"], args, device, save_path)
        val_loss_all.append(val_loss)
        val_acc_all.append(val_topk[1].mean())
        if val_loss < best_test_loss - args.min_delta:
            best_test_loss = val_loss
            torch.save(student_model.state_dict(), os.path.join(save_path, "model_best.pth"))
            save_checkpoint({"epoch": epoch + 1, "state_dict": student_model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "test_loss": val_loss}, save_path, filename="Final_model.pth")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if args.use_early_stopping and epochs_without_improvement >= args.patience:
                break
    return train_acc_all, train_loss_all, val_acc_all, val_loss_all, lrs, train_task_loss_all


def validate_model(epoch, model, dataloader, args, device, save_path):
    """Per-epoch validation on the test loader (image-only, matches training)."""
    val_loss, topk_acc, dba_score = _aggregate_eval_metrics(
        model, dataloader, args, device, show_progress=False
    )
    param_info = f" (seq_len={args.seq_length}, num_pred={args.num_pred})"
    _print_eval_summary(f"Epoch {epoch} Test Loss{param_info}", val_loss, topk_acc, dba_score)

    # Save results to training_results.txt (append each epoch)
    _append_results_txt(
        save_path,
        title=f"Epoch {epoch} Results Summary{param_info}",
        loss=val_loss,
        topk_acc=topk_acc,
        dba_score=dba_score,
        filename="training_results.txt",
    )

    return val_loss, topk_acc, dba_score


def test_model(model, dataloader, args, device, save_path):
    """Final evaluation after training (same protocol as validate_model, with progress bar)."""
    val_loss, topk_acc, dba_score = _aggregate_eval_metrics(
        model, dataloader, args, device, show_progress=True, desc="Test"
    )
    _print_eval_summary("Test Loss", val_loss, topk_acc, dba_score)

    # Save results to training_results.txt
    _append_results_txt(
        save_path,
        title="Test Results Summary",
        loss=val_loss,
        topk_acc=topk_acc,
        dba_score=dba_score,
        filename="training_results.txt",
    )

    return val_loss, topk_acc, dba_score


def main():
    args = parse_args()
    set_seed(args.seed)
    current_dir = os.path.dirname(__file__)
    data_root = current_dir + "/dataset/"
    train_dir = os.path.join(data_root, args.train_csv_name)
    test_dir = os.path.join(data_root, args.test_csv_name)
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    print(f"=====DEVICE: {device}=====")
    if torch.cuda.is_available() and select_best_gpu() != "":
        torch.cuda.set_device(int(select_best_gpu()))

    # Log debug mode status
    if args.debug:
        print(f"=====DEBUG MODE ENABLED - Saving to debug folder=====")
        # Override some settings for faster debugging
        if args.epochs > 5:
            args.epochs = 3
            print(f"=====DEBUG MODE: Reducing epochs to {args.epochs} for faster testing=====")
        if args.dataset_pct > 0.2:
            args.dataset_pct = 0.1
            print(f"=====DEBUG MODE: Reducing dataset to {args.dataset_pct*100}% for faster testing=====")
    

    gru_params = (args.feature_size, args.gru_hidden_size, args.gru_num_layers)
    if args.model_name == "DirectStem16Image_v2":
        model = DirectStem16Image_v2(
            feature_size=args.feature_size,
            num_classes=args.num_classes,
            gru_params=gru_params,
            num_heads=args.num_heads,
            ds_conv_block_version=args.ds_conv_block_version,
        )
    elif args.model_name == "ImageModalityNet_MHA":
        model = ImageModalityNet_MHA(
            feature_size=args.feature_size,
            num_classes=args.num_classes,
            gru_params=gru_params,
            num_heads=args.num_heads,
        )
    else:
        model = ImageOnlyAblationFusionNet(
            feature_size=args.feature_size,
            num_classes=args.num_classes,
            gru_params=gru_params,
            num_heads=args.num_heads,
        )
    model = model.to(device)

    day = datetime.datetime.now().strftime("%m-%d-%Y")
    hm = datetime.datetime.now().strftime("%H_%M")
    base_save = "saved_folder_debug" if args.debug else args.save_dir
    save_directory = os.path.join(current_dir, base_save, f"{model.name}_DataAug{args.augment_data}_LabelSmooth{args.label_smoothing}_{day}_{hm}{'_DEBUG' if args.debug else ''}")
    os.makedirs(save_directory, exist_ok=True)

    img_resize = transf.Resize((224, 224))
    proc_pipe = transf.Compose([transf.ToPILImage(), img_resize])
    train_loader = DataLoader(DataFeed(data_root, train_dir, args.seq_length, transform=proc_pipe, portion=args.dataset_pct, beam_soft_label_temperature=args.beam_soft_label_temperature, full_beam_dim=args.full_beam_dim), batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(DataFeed(data_root, test_dir, args.seq_length, transform=proc_pipe, portion=args.dataset_pct, beam_soft_label_temperature=args.beam_soft_label_temperature, full_beam_dim=args.full_beam_dim), batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers)
    dataloaders = {"train": train_loader, "test": test_loader}

    with open(os.path.join(save_directory, "params.txt"), "w") as f:
        json.dump(args.__dict__, f, indent=2)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for module in sys.modules.values():
        p = getattr(module, "__file__", None)
        if p and os.path.abspath(p).endswith(".py") and os.path.dirname(os.path.abspath(p)) == script_dir:
            dst = os.path.join(save_directory, os.path.basename(p))
            if os.path.abspath(p) != os.path.abspath(dst):
                shutil.copy2(p, dst)

    with open(os.path.join(save_directory, "params.txt"), "a") as f:
        old_stdout = sys.stdout
        sys.stdout = buffer = std_io.StringIO()
        image_input = torch.randn(1, args.seq_length - 1, 1, 224, 224).to(device)
        try:
            print(summary(model, image_input, None, show_input=True, show_hierarchical=True))
        except Exception:
            print("Model summary could not be generated")
        sys.stdout = old_stdout
        f.write(buffer.getvalue())
        
        model_flops, model_params = compute_flops(model, image_input, "Model")
        # Calculate and write parameter information
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())

        
        f.write(f"\nModel Parameters:\n")
        f.write(f"Total parameters: {total_params:,}\n")
        f.write(f"Trainable parameters: {trainable_params:,}\n")
        f.write(f"Non-trainable parameters: {total_params - trainable_params:,}\n")
        f.write(f'TrainDataSize: {len(train_loader.dataset)}\n')
        f.write(f'TestDataSize: {len(test_loader.dataset)}\n')
        f.write(f"Label smoothing: {args.label_smoothing}\n")
        f.write(f"Hard label weight: {1 - args.soft_label_weight if args.label_smoothing else 1.0}\n")
        f.write(f"Soft label weight: {args.soft_label_weight if args.label_smoothing else 0.0}\n")
        f.write(f"Model FLOPs: {model_flops/1e6:.3f} MFLOPs, {model_params/1e6:.3f} M params\n")

        f.write(f"\nTraining Mode:\n")
        f.write(f"Debug mode: {args.debug}\n")
        f.write(f"Save directory: {save_directory}\n")

    print(f"Total trainable parameters in model: {trainable_params:,}")

    if args.debug:
        print(f"=====DEBUG MODE: All outputs saved to {save_directory}=====")
    else:
        print(f"=====TRAINING MODE: All outputs saved to {save_directory}=====")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.T_0, T_mult=args.T_mult, eta_min=args.eta_min)
    print(f"Using cosine annealing warm restarts scheduler with T_0={args.T_0}, T_mult={args.T_mult}, eta_min={args.eta_min}")
    start = time.time()
    train_acc, train_loss, test_acc, test_loss, lrs, train_task_loss = train_model(model, dataloaders, args, optimizer, scheduler, device, save_directory)
    np.savez(os.path.join(save_directory, "training_outputs.npz"), train_acc_hist=np.array(train_acc), train_loss_hist=np.array(train_loss), test_acc_hist=np.array(test_acc), test_loss_hist=np.array(test_loss), learning_rates=np.array(lrs), train_task_loss_hist=np.array(train_task_loss))
    time_elapsed = time.time() - start
    with open(os.path.join(save_directory, 'training_results.txt'), "a") as f:
        f.write(f'Finished Training with DataAug{args.augment_data}_LabelSmooth{args.label_smoothing}\n')
        f.write(f'Training completed in {time_elapsed // 60}m {time_elapsed % 60}s\n')

    # Test the model
    print(f"Testing the model...")
    model.load_state_dict(torch.load(os.path.join(save_directory, "model_best.pth")))
    test_model(model, test_loader, args, device, save_directory)
    plot_training_curves(train_acc, train_loss, test_acc, test_loss, lrs, save_directory, train_task_loss_hist=train_task_loss)
    print(f"Training completed in {((time.time()-start)//60):.0f}m {((time.time()-start)%60):.0f}s")


if __name__ == "__main__":
    main()
