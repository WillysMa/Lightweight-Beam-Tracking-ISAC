#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm


def _build_resnet18(in_channels: int, pretrained: bool):
    if pretrained:
        try:
            from torchvision.models import ResNet18_Weights

            rnet = tvm.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        except Exception:
            rnet = tvm.resnet18(pretrained=True)
    else:
        try:
            rnet = tvm.resnet18(weights=None)
        except Exception:
            rnet = tvm.resnet18(pretrained=False)

    if in_channels != 3:
        old = rnet.conv1
        new = nn.Conv2d(
            in_channels,
            old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=False,
        )
        with torch.no_grad():
            w_avg = old.weight.mean(dim=1, keepdim=True)
            new.weight.copy_(w_avg.repeat(1, in_channels, 1, 1) * (3.0 / in_channels))
        rnet.conv1 = new
    return rnet


class ResNet18ModalityFeatureExtractor(nn.Module):
    """Per-frame ResNet-18 features: backbone + global avg pool + linear head."""

    def __init__(self, n_feature: int, in_channels: int = 1, pretrained: bool = True):
        super().__init__()
        rnet = _build_resnet18(in_channels, pretrained)
        self.backbone = nn.Sequential(
            rnet.conv1,
            rnet.bn1,
            rnet.relu,
            rnet.maxpool,
            rnet.layer1,
            rnet.layer2,
            rnet.layer3,
            rnet.layer4,
            rnet.avgpool,
        )
        self.head = nn.Linear(512, n_feature)

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.reshape(b * t, c, h, w)
        z = self.backbone(x)
        z = torch.flatten(z, 1)
        z = self.head(z)
        return z.view(b, t, -1)


class ImageOnlyAblationFusionNet(nn.Module):
    """
    Image-only ResNet-18 baseline (encoder + fusion + GRU + MHA + classifier).
    """

    def __init__(
        self,
        feature_size: int,
        num_classes: int,
        gru_params,
        image_channels: int = 1,
        num_heads: int = 8,
        resnet_pretrained: bool = True,
        gru_dropout: float = 0.5,
    ):
        super().__init__()
        gru_input_size, gru_hidden_size, gru_num_layers = gru_params
        assert gru_input_size == feature_size, (
            f"gru_input_size ({gru_input_size}) must equal feature_size ({feature_size})"
        )
        self.name = "ImageOnlyAblationFusionNet"

        self.image_encoder = ResNet18ModalityFeatureExtractor(
            n_feature=feature_size,
            in_channels=image_channels,
            pretrained=resnet_pretrained,
        )
        self.fusion = nn.Linear(feature_size, gru_input_size)
        self.norm_pre = nn.LayerNorm(gru_input_size)

        self.gru = nn.GRU(
            input_size=gru_input_size,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            dropout=gru_dropout if gru_num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=gru_hidden_size,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, image_batch, radar_batch=None, beam=None):
        x = self.image_encoder(image_batch)
        x = self.fusion(x)
        features = self.norm_pre(x)

        seq_out, _ = self.gru(features)
        attn_out, _ = self.multihead_attention(seq_out, seq_out, seq_out)
        enhanced_seq_out = attn_out + seq_out

        pred = self.classifier(enhanced_seq_out)
        return pred, features, enhanced_seq_out
