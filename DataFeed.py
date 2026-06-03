#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from skimage import io
from skimage.color import rgb2gray
from scipy.ndimage import gaussian_filter


def repair_nonfinite_beam_powers_with_neighbors(p):
    p = np.asarray(p, dtype=np.float64).copy()
    bad = ~np.isfinite(p)
    if not bad.any():
        return p
    n = p.size
    left = np.full(n, np.nan, dtype=np.float64)
    right = np.full(n, np.nan, dtype=np.float64)
    prev = np.nan
    for i in range(n):
        if np.isfinite(p[i]):
            prev = p[i]
        left[i] = prev
    prev = np.nan
    for i in range(n - 1, -1, -1):
        if np.isfinite(p[i]):
            prev = p[i]
        right[i] = prev
    for i in np.flatnonzero(bad):
        if np.isfinite(left[i]) and np.isfinite(right[i]):
            p[i] = 0.5 * (left[i] + right[i])
        elif np.isfinite(left[i]):
            p[i] = left[i]
        elif np.isfinite(right[i]):
            p[i] = right[i]
        else:
            p[i] = 0.0
    return p


def beam_powers_txt_to_soft_label(path, full_dim=64, temperature=1.0, eps=1e-12):
    p = np.atleast_1d(np.loadtxt(path, dtype=np.float64)).ravel()
    if p.size != full_dim:
        raise ValueError(f"Expected {full_dim} beam powers in {path!r}, got {p.size}")
    p = repair_nonfinite_beam_powers_with_neighbors(p)
    p = np.maximum(p, 0.0).astype(np.float64)
    if np.sum(p) <= 0.0:
        return np.full(full_dim, 1.0 / full_dim, dtype=np.float32)
    t = max(float(temperature), eps)
    logits = np.log(p + eps) / t
    logits -= np.max(logits)
    probs = np.exp(logits)
    probs /= np.sum(probs)
    return probs.astype(np.float32)


def pool_beam_soft_for_downsample(soft, downsample_ratio, num_classes, full_dim=64):
    soft = torch.nan_to_num(soft, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)
    n_cls = max(int(num_classes), 1)

    def _renorm(x):
        x = x[..., :n_cls]
        row_sum = x.sum(dim=-1, keepdim=True)
        out = x / row_sum.clamp(min=1e-8)
        bad = row_sum.squeeze(-1) < 1e-12
        if bad.any():
            out = torch.where(
                bad.unsqueeze(-1).expand_as(out),
                torch.full_like(out, 1.0 / n_cls),
                out,
            )
        return out

    if downsample_ratio <= 1:
        return _renorm(soft)
    if soft.shape[-1] != full_dim:
        raise ValueError(f"soft last dim {soft.shape[-1]} != full_dim {full_dim}")
    pad = (-full_dim) % downsample_ratio
    if pad:
        soft = F.pad(soft, (0, pad))
    pooled = soft.reshape(*soft.shape[:-1], soft.shape[-1] // downsample_ratio, downsample_ratio).sum(-1)
    return _renorm(pooled)


def create_samples(root_csv, portion=1.0):
    df = pd.read_csv(root_csv, na_values="").fillna(-99)
    num_data = int(len(df) * portion)
    data_samples_rgb = []
    pred_beam = []
    inp_beam = []
    future_beam_cols = [col for col in df.columns if col.startswith("future_beam")]
    future_beam_cols.sort()
    for _, row in df.head(num_data).iterrows():
        data_samples_rgb.append(row["camera1":"camera8"].tolist())
        pred_beam.append(row[future_beam_cols].tolist())
        inp_beam.append(row["beam1":"beam8"].tolist())
    return data_samples_rgb, inp_beam, pred_beam


class DataFeed(Dataset):
    def __init__(
        self,
        data_root,
        root_csv,
        seq_len,
        transform=None,
        portion=1.0,
        beam_soft_label_temperature=1.0,
        full_beam_dim=64,
    ):
        self.data_root = data_root
        self.samples_rgb, self.inp_val, self.pred_val = create_samples(root_csv, portion=portion)
        self.seq_len = seq_len
        self.transform = transform
        self.beam_soft_label_temperature = beam_soft_label_temperature
        self.full_beam_dim = full_beam_dim

    def __len__(self):
        return len(self.samples_rgb)

    def __getitem__(self, idx):
        samples_rgb = self.samples_rgb[idx]
        beam_val = self.pred_val[idx]
        inp_beam_paths = self.inp_val[idx]

        image_val = np.zeros((self.seq_len, 224, 224), dtype=np.float32)
        image_motion_masks = np.zeros((self.seq_len - 1, 224, 224), dtype=np.float32)
        beam_past = []

        def _p(rel_path):
            return os.path.join(self.data_root, rel_path.lstrip("/"))

        for i, smp_rgb_path in enumerate(samples_rgb):
            inp_txt = _p(inp_beam_paths[i])
            powers = np.maximum(np.atleast_1d(np.loadtxt(inp_txt, dtype=np.float64)).ravel(), 0.0)
            beam_past.append(int(np.argmax(powers)))

            img = self.transform(io.imread(_p(smp_rgb_path)))
            img = gaussian_filter(rgb2gray(img), sigma=1)
            image_val[i, ...] = img

            if i >= 1:
                diff = np.abs(image_val[i, ...] - image_val[i - 1, ...])
                threshold = 0.1 * np.max(diff)
                image_motion_masks[i - 1, ...] = (diff > threshold).astype(np.float32)

        beam_future = []
        for beam_path in beam_val:
            fut_powers = np.atleast_1d(np.loadtxt(_p(beam_path), dtype=np.float64)).ravel()
            fut_powers = np.maximum(repair_nonfinite_beam_powers_with_neighbors(fut_powers), 0.0)
            beam_future.append(int(np.argmax(fut_powers)))

        soft_rows = [
            beam_powers_txt_to_soft_label(
                _p(inp_beam_paths[-1]),
                full_dim=self.full_beam_dim,
                temperature=self.beam_soft_label_temperature,
            )
        ]
        for fp in beam_val:
            soft_rows.append(
                beam_powers_txt_to_soft_label(
                    _p(fp),
                    full_dim=self.full_beam_dim,
                    temperature=self.beam_soft_label_temperature,
                )
            )

        return (
            torch.tensor(image_motion_masks, dtype=torch.float32),
            torch.tensor(beam_past, dtype=torch.int64),
            torch.squeeze(torch.tensor(beam_future, dtype=torch.int64)),
            torch.from_numpy(np.stack(soft_rows, axis=0)),
        )
