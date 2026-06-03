#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Mengyuan Ma
@contact:mamengyuan410@gmail.com
@file: test_model.py   
@time: 2026/06/03 17:50
"""
import os

if os.name == "nt":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import shutil
import time 
import subprocess
import json
import argparse
from torch.utils.data import DataLoader
from pytorch_model_summary import summary
from tqdm import tqdm
import sys
import datetime
import torchvision.transforms as transf
import matplotlib.pyplot as plt
from thop import profile as thop_profile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from DataFeed import DataFeed
from MyFunc import *
from model_icassp import ImageModalityNet_MHA
from model_pp import DirectStem16Image_v2
from model_ResNet import ImageOnlyAblationFusionNet

'''
Note that 
- The number of GRU layers is 1 for DirectStem16Image_v2, 2 for others
- Proposed model is DirectStem16Image_v2
- CNN_GRU model is ImageModalityNet_MHA
- ResNet model is ImageOnlyAblationFusionNet
'''

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Model Testing')
    
    # testing parameters
    parser.add_argument('--test_batch_size', type=int, default=32, help='Test batch size')
    parser.add_argument('--test_csv_name', type=str, default='test_seqs_RA.csv', help='Test csv name')
    parser.add_argument('--loss_type', type=str, default='focal', choices=['crossentropy', 'focal'], 
                        help='Loss function type')
    parser.add_argument(
        '--model_arch',
        type=str,
        default='ImageOnlyAblationFusionNet',
        choices=['DirectStem16Image_v2', 'ImageModalityNet_MHA', 'ImageOnlyAblationFusionNet'],
        help='Model architecture to test',
    )
    parser.add_argument('--checkpoint', type=str, default='./All_models/model_ImageOnlyAblationFusionNet_AugDataFalse_LabelSmoothTrue.pth', help='Weights file (.pth)')
    parser.add_argument('--data_root', type=str, default='./dataset/', help='Data root directory')
    parser.add_argument('--feature_size', type=int, default=64, help='Feature size')
    parser.add_argument('--gru_hidden_size', type=int, default=64, help='GRU hidden size')
    parser.add_argument('--gru_num_layers', type=int, default=2, help='Number of GRU layers') # 1 for DirectStem16Image_v2, 2 for others
    parser.add_argument('--num_classes', type=int, default=64, help='Number of classes')
    parser.add_argument('--seq_length', type=int, default=8, help='DataFeed sequence length')
    parser.add_argument('--num_pred', type=int, default=3, help='Number of predictions')
    parser.add_argument('--downsample_ratio', type=int, default=1, help='Downsample ratio')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of heads for multi-head attention')
    parser.add_argument('--ds_conv_block_version', type=int, default=2, help='DS block version (DirectStem16Image_v2)')

    # Testing setting
    parser.add_argument('--use_gpu', action='store_true', default=True, help='Use GPU if available')
    parser.add_argument('--debug', action='store_true', default=False, help='Enable debug mode (saves to saved_folder_debug)')
    parser.add_argument('--dataset_pct', type=float, default=1.0, help='Dataset percentage to use')
    parser.add_argument('--save_dir', type=str, default='saved_folder_test', help='Save directory')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')

    return parser.parse_args()

def get_free_gpu():
    import subprocess
    result = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits"
        ]
    )

    lines = result.decode().strip().split("\n")

    best_gpu = min(lines, key=lambda x: int(x.split(",")[1]))
    return best_gpu.split(",")[0]

def build_model(args):
    gru_params = (args.feature_size, args.gru_hidden_size, args.gru_num_layers)
    if args.model_arch == 'DirectStem16Image_v2':
        return DirectStem16Image_v2(
            args.feature_size, args.num_classes, gru_params,
            num_heads=args.num_heads, ds_conv_block_version=args.ds_conv_block_version,
        )
    if args.model_arch == 'ImageModalityNet_MHA':
        return ImageModalityNet_MHA(
            args.feature_size, args.num_classes, gru_params, num_heads=args.num_heads,
        )
    return ImageOnlyAblationFusionNet(
        args.feature_size, args.num_classes, gru_params, num_heads=args.num_heads,
    )


def _build_image_batch(image_masks, args, device):
    img = image_masks.unsqueeze(2)
    b, _, c, h, w = img.shape
    return torch.cat([img, torch.zeros(b, args.num_pred, c, h, w, device=img.device)], dim=1).to(device)


def compute_flops(model, inputs, name, batch_size=1):
    if thop_profile is None:
        print(f"[FLOPs] thop not available; skip {name} FLOPs.")
        return None, None
    if not isinstance(inputs, (list, tuple)):
        inputs = (inputs,)
    model.eval()
    with torch.no_grad():
        flops, params = thop_profile(model, inputs=inputs, verbose=False)
    print(f"[Per-sample FLOPs] {name}: {flops/batch_size/1e6:.3f} MFLOPs, {params/1e6:.3f} M params")
    return flops, params
    


def run_evaluation(model, dataloader, args, device):
    """Image-only evaluation (same batching as train_agumt)."""
    model.eval()
    if args.loss_type == 'focal':
        criterion = FocalLoss(alpha=1, gamma=2)
    else:
        criterion = nn.CrossEntropyLoss()

    val_loss = 0.0
    all_outputs = []
    all_labels = []

    with tqdm(dataloader, unit="batch", file=sys.stdout) as tepoch:
        for i, batch in enumerate(tepoch, 0):
            tepoch.set_description(f"Testing batch {i}")
            image_masks, beam, label, _ = batch[:4]
            beam_down = torch.floor(beam.float() / args.downsample_ratio).to(torch.int64)
            label_down = torch.floor(label.float() / args.downsample_ratio).to(torch.int64)
            test_label = torch.cat(
                [beam_down[..., -1:], label_down[:, : args.num_pred]], dim=-1
            ).to(device)

            with torch.no_grad():
                outputs, _, _ = model(_build_image_batch(image_masks, args, device), None)
            outputs = outputs[:, -(args.num_pred + 1) :, :]
            val_loss += criterion(
                outputs.reshape(-1, args.num_classes), test_label.flatten()
            ).item()
            all_outputs.append(outputs)
            all_labels.append(test_label)

    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    topk_acc, total = calculate_topk_accuracy(all_outputs, all_labels)
    dba_score = calculate_dba_score(all_outputs, all_labels)
    val_loss /= max(len(dataloader), 1)
    
    print("DBA-Score (Top-3):", dba_score)
    print('Top-K Accuracy:', flush=True)
    for k, acc in topk_acc.items():
        print(f'Top-{k}: {acc}', flush=True)
    

    return val_loss, topk_acc, dba_score


def main():
    """Main function for testing only"""
    args = parse_args()

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = args.data_root
    test_dir = os.path.join(data_root, args.test_csv_name)

    # Data preprocessing
    img_resize = transf.Resize((224, 224))
    proc_pipe = transf.Compose([transf.ToPILImage(), img_resize])

    test_dataset = DataFeed(
        data_root, test_dir, args.seq_length, transform=proc_pipe, portion=args.dataset_pct
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers
    )

    print(f'TestDataSize: {len(test_loader.dataset)}')

    # Setup device
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    model = build_model(args)
    model_path = args.checkpoint if os.path.isabs(args.checkpoint) else os.path.join(current_dir, args.checkpoint)
    state_dict = torch.load(model_path, map_location=device)
    # if isinstance(state_dict, dict) and "state_dict" in state_dict:
    #     state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    print(f"===== {args.model_arch} loaded from: {model_path} =====")
    try:
        t_in = args.seq_length - 1
        image_input = torch.randn(1, t_in, 1, 224, 224).to(device)
        compute_flops(model, (image_input, None), args.model_arch)
    except Exception as exc:
        print(f"Warning: FLOPs/params computation failed: {exc}", flush=True)

    print('\nStart testing model...\n', flush=True)
    run_evaluation(model, test_loader, args, device)

   

if __name__ == "__main__":
    main()


