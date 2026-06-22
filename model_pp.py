#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Straightforward, self-contained re-implementation of ``DirectStem16Image_v2``.

Architecture (image-only):
    ResNet-like stem (1->16) -> 4 MobileNetV2 inverted-residual blocks
    (16->32->64->128->256, each stride 2) -> global max pool ->
    Linear projection -> LayerNorm -> GRU -> self multi-head attention (residual)
    -> classifier.

"""

import torch
import torch.nn as nn
import torchvision.models as tvm


class InvertedResidualV2(nn.Module):
    """MobileNetV2 block: 1x1 expand -> 3x3 depthwise -> 1x1 linear project."""

    def __init__(self, in_ch, out_ch, stride=1, expand_ratio=2):
        super().__init__()
        assert stride in (1, 2)
        hidden = int(round(in_ch * expand_ratio))
        self.use_residual = stride == 1 and in_ch == out_ch

        layers = []
        if expand_ratio != 1:
            layers += [
                nn.Conv2d(in_ch, hidden, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU6(inplace=True),
            ]
        else:
            hidden = in_ch
        layers += [
            nn.Conv2d(hidden, hidden, kernel_size=3, stride=stride, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.block(x) if self.use_residual else self.block(x)


def _make_stem(in_channels: int, out_channels: int, pretrained: bool) -> nn.Sequential:
    """ResNet-18 stem geometry (7x7 s2 -> BN -> ReLU -> 3x3 maxpool s2).

    When ``pretrained`` is True, conv/bn weights are sliced from ImageNet ResNet-18
    (requires ``out_channels <= 64``). For ``in_channels != 3`` the RGB filters are
    averaged across the channel dim.
    """
    conv = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False)
    bn = nn.BatchNorm2d(out_channels)

    if pretrained:
        assert out_channels <= 64, "pretrained stem slicing needs out_channels <= 64"
        try:
            from torchvision.models import ResNet18_Weights
            rnet = tvm.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        except Exception:
            rnet = tvm.resnet18(pretrained=True)

        w = rnet.conv1.weight  # [64, 3, 7, 7]
        if in_channels != 3:
            w = w.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1) * (3.0 / in_channels)
        with torch.no_grad():
            conv.weight.copy_(w[:out_channels])
            bn.weight.copy_(rnet.bn1.weight[:out_channels])
            bn.bias.copy_(rnet.bn1.bias[:out_channels])
            bn.running_mean.copy_(rnet.bn1.running_mean[:out_channels])
            bn.running_var.copy_(rnet.bn1.running_var[:out_channels])

    return nn.Sequential(conv, bn, nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))


class DirectStem16Image_v2(nn.Module):
    """Image-only beam predictor (simplified, equivalent to model_pp version)."""

    CHANNELS = (16, 32, 64, 128, 256)  # stem out + 4 block outputs

    def __init__(
        self,
        feature_size,
        num_classes,
        gru_params,
        image_channels=1,
        num_heads=8,
        ds_conv_block_version=2,
        resnet_pretrained=True,
    ):
        super().__init__()
        gru_input_size, gru_hidden_size, gru_num_layers = gru_params
        assert gru_input_size == feature_size, (
            f"gru_input_size ({gru_input_size}) must equal feature_size ({feature_size})"
        )
        assert ds_conv_block_version == 2, "model_pp_new only implements the default MobileNetV2 block"
        self.name = "DirectStem16Image_v2"

        chans = self.CHANNELS
        self.stem = _make_stem(image_channels, chans[0], resnet_pretrained)
        self.body = nn.Sequential(*[
            InvertedResidualV2(chans[i], chans[i + 1], stride=2, expand_ratio=2)
            for i in range(len(chans) - 1)
        ])
        self.image_global_max_pool = nn.AdaptiveMaxPool2d(1)

        self.fusion_layer = nn.Sequential(
            nn.Linear(chans[-1], 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, feature_size),
        )

        self.layer_norm = nn.LayerNorm(gru_input_size)
        self.GRU = nn.GRU(
            input_size=gru_input_size,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            dropout=0.5 if gru_num_layers > 1 else 0,
            batch_first=True,
        )
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=gru_hidden_size, num_heads=num_heads, dropout=0.1, batch_first=True
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
        b, t, c, h, w = image_batch.shape
        x = image_batch.reshape(b * t, c, h, w)

        x = self.stem(x)
        x = self.body(x)
        x = self.image_global_max_pool(x).flatten(1)        # (b*t, 256)

        features = self.fusion_layer(x).view(b, t, -1)      # (b, t, feature_size)
        features = self.layer_norm(features)

        seq_out, _ = self.GRU(features)
        attn_out, _ = self.multihead_attention(seq_out, seq_out, seq_out)
        enhanced_seq_out = attn_out + seq_out

        pred = self.classifier(enhanced_seq_out)
        return pred, features, enhanced_seq_out
