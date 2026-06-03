#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Mengyuan Ma
@contact:mamengyuan410@gmail.coms
@file: MyFunc.py
@time: 2026/06/03 17:49
"""
import numpy as np
import pandas as pd
import torch
from skimage import io
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transf
import matplotlib.pyplot as plt
import os
from skimage.color import rgb2gray
from scipy.ndimage import gaussian_filter
from scipy.io import loadmat
from torch.optim.lr_scheduler import _LRScheduler
from collections.abc import Iterable
import torch.nn as nn
import torch.nn.functional as F
from math import log, cos, pi, floor
import random
from thop import profile as thop_profile

def set_seed(seed=0):
    """Set all random seeds for reproducible training"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # For deterministic behavior
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def select_best_gpu():
    if not torch.cuda.is_available():
        return ""

    # Prefer the GPU with the most *global* free memory (includes other users' processes),
    # not torch.cuda.memory_allocated(i), which is only this process and is ~0 at startup.
    best_gpu = 0
    max_free = -1
    for i in range(torch.cuda.device_count()):
        try:
            free_b, _total_b = torch.cuda.mem_get_info(i)
        except Exception:
            torch.cuda.set_device(i)
            free_b = -torch.cuda.memory_allocated(i)
        if free_b > max_free:
            max_free = free_b
            best_gpu = i

    return str(best_gpu)


def compute_flops(model, inputs, name, batch_size=1):
    if thop_profile is None:
        print(f"[FLOPs] thop not available; skip {name} FLOPs.")
        return None, None
    # thop.profile() calls model(*inputs); a single tensor must be passed as (tensor,) so the model gets one argument
    if not isinstance(inputs, (list, tuple)):
        inputs = (inputs,)
    model.eval()
    with torch.no_grad():
        flops, params = thop_profile(model, inputs=inputs, verbose=False)
    print(f"[FLOPs] {name}: {flops/batch_size/1e6:.3f} M FLOPs, {params/1e6:.3f} M params")
    return flops, params



def calculate_weight_update_interval(epoch, initial_interval=20, min_interval=3, beta=0.8, start_epoch=10):
    """
    Calculate the interval for weight updates based on current epoch.
    Starts weight updates at specified epoch, then decreases interval by beta factor after each update cycle.
    
    Args:
        epoch: Current epoch number
        initial_interval: Initial interval between weight updates (default: 20)
        min_interval: Minimum interval between weight updates (default: 3)
        beta: Time decay coefficient applied to reduce interval (default: 0.8)
        start_epoch: Epoch to start weight updates (default: 20)
    
    Returns:
        current_interval: Current interval for weight updates
                         Returns -1 if before start_epoch (indicating no updates)
    """
    # No weight updates before start_epoch
    if epoch < start_epoch:
        return -1
    
    # Calculate which interval generation we're in
    epochs_since_start = epoch - start_epoch
    current_interval = initial_interval
    accumulated_epochs = 0
    
    # Find the current interval by simulating the decay process
    while accumulated_epochs + current_interval <= epochs_since_start:
        accumulated_epochs += current_interval
        # Apply decay coefficient to get next interval
        current_interval = int(current_interval * beta)
        # Ensure minimum interval
        current_interval = max(current_interval, min_interval)
    
    return current_interval

    
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean', ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs,
            targets,
            reduction='none',
            ignore_index=self.ignore_index,
        )
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def calculate_topk_accuracy(outputs, labels, k_values=[1, 2, 3, 5, 10]):
    """Calculate top-k accuracy for given k values"""
    num_pred = labels.shape[1]
    topk_correct = {k: np.zeros((num_pred,)) for k in k_values}
    total = torch.sum(labels != -100, dim=0).cpu().numpy()
    
    _, idx = torch.topk(outputs, max(k_values), dim=-1)
    idx = idx.cpu().numpy()
    labels = labels.cpu().numpy()
    
    for i in range(labels.shape[1]):  # for each time step
        for j in range(labels.shape[0]):  # examine all samples
            for k in k_values:
                topk_correct[k][i] += np.isin(labels[j, i], idx[j, i, :k])
    
    # Calculate accuracy
    topk_acc = {}
    for k in k_values:
        topk_acc[k] = topk_correct[k] / (total + 1e-8)  # Add small epsilon to avoid division by zero
    
    return topk_acc, total

def calculate_dba_score(outputs, labels, delta=5):
    """Calculate DBA (Distance-Based Accuracy) score"""
    num_pred = labels.shape[1]
    dba_score = np.zeros((num_pred,))
    valid_count = np.zeros((num_pred,))
    
    _, idx = torch.topk(outputs, 3, dim=-1)  # top-3 predictions for DBA
    idx = idx.cpu().numpy()
    labels = labels.cpu().numpy()
    
    for t in range(labels.shape[1]):
        for b in range(labels.shape[0]):
            gt = labels[b, t]
            if gt == -100:
                continue  # skip invalid label
            
            preds = idx[b, t, :3]  # top-3 predictions
            norm_dists = np.minimum(np.abs(preds - gt) / delta, 1.0)
            min_norm_dist = np.min(norm_dists)
            
            dba_score[t] += min_norm_dist
            valid_count[t] += 1
    
    # Avoid division by zero
    valid_count[valid_count == 0] = 1
    dba_score = 1 - (dba_score / valid_count)
    
    return dba_score

def save_checkpoint(state, save_path, filename='checkpoint.pth'):
    """Save training checkpoint"""
    filepath = os.path.join(save_path, filename)
    torch.save(state, filepath)
    print(f"Checkpoint saved to {filepath}")

def load_checkpoint(save_path, model, optimizer=None, scheduler=None):
    """Load training checkpoint"""
    checkpoint_path = os.path.join(save_path, 'Final_model.pth')
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        
        if optimizer is not None and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        
        if scheduler is not None and 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        
        print(f"Loaded checkpoint '{checkpoint_path}' (epoch {checkpoint['epoch']})")
        return start_epoch, checkpoint.get('test_loss', 0.0)
    else:
        print(f"No checkpoint found at '{checkpoint_path}'")
        return 0, 0.0
    

    

def plot_training_curves(
    train_acc_hist,
    train_loss_hist,
    test_acc_hist,
    test_loss_hist,
    lrs,
    save_path,
    train_task_loss_hist=None,
):
    """Plot and save standard image-only training curves."""
    epochs = len(train_acc_hist)
    
    # Learning rate schedule
    plt.figure()
    plt.plot(np.arange(1, epochs + 1), lrs)
    plt.xlabel('Epoch')
    plt.ylabel('Learning rate')
    plt.grid(True)
    plt.title('Learning Rate Schedule')
    plt.savefig(os.path.join(save_path, 'LR_schedule.png'))
    plt.close()
    
    # Accuracy curves
    plt.figure()
    plt.plot(np.arange(1, epochs + 1), train_acc_hist, '-o', label='Train')
    plt.plot(np.arange(1, epochs + 1), test_acc_hist, '-o', label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Train vs Test Accuracy')
    plt.grid(True)
    plt.savefig(os.path.join(save_path, 'Accuracy_curves.png'))
    plt.close()
    
    # Loss curves
    plt.figure()
    plt.plot(np.arange(1, epochs + 1), train_loss_hist, '-o', label='Train Total')
    plt.plot(np.arange(1, epochs + 1), test_loss_hist, '-o', label='Test')
    
    # Add optional task-only loss component if available
    if train_task_loss_hist is not None:
        plt.plot(np.arange(1, epochs + 1), train_task_loss_hist, '--', label='Train Task Loss')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss Curves')
    plt.grid(True)
    plt.savefig(os.path.join(save_path, 'Loss_curves.png'))
    plt.close()



