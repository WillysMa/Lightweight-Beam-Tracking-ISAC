
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torchvision.models as tvm


def _adapt_resnet18_conv1_for_channels(backbone, in_channels: int, pretrained: bool) -> None:
    old = backbone.conv1
    if in_channels == 3:
        return
    new = nn.Conv2d(
        in_channels,
        old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        bias=False,
    )
    if pretrained and old.weight.shape[1] == 3:
        with torch.no_grad():
            w_avg = old.weight.mean(dim=1, keepdim=True)
            new.weight.copy_(w_avg.repeat(1, in_channels, 1, 1) * (3.0 / in_channels))
    else:
        nn.init.kaiming_normal_(new.weight, mode="fan_out", nonlinearity="relu")
    backbone.conv1 = new


class InvertedResidualMobileNetV2(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, expand_ratio=6):
        super().__init__()
        assert stride in (1, 2)
        hidden = int(round(in_ch * expand_ratio))
        self.use_residual = stride == 1 and in_ch == out_ch
        layers = []
        if expand_ratio != 1:
            layers += [
                nn.Conv2d(in_ch, hidden, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU6(inplace=True),
            ]
        else:
            hidden = in_ch
        layers += [
            nn.Conv2d(hidden, hidden, kernel_size=3, stride=stride, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.block(x)
        return self.block(x)


class SqueezeExciteMobileNetV3(nn.Module):
    def __init__(self, channels: int, se_ratio: int = 4):
        super().__init__()
        hidden = max(8, channels // se_ratio)
        self.op = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            nn.Hardsigmoid(),
        )

    def forward(self, x):
        return x * self.op(x)


class InvertedResidualMobileNetV3(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        expand_ratio: int = 6,
        use_se: bool = True,
        se_ratio: int = 4,
        kernel_size: int = 3,
    ):
        super().__init__()
        assert stride in (1, 2)
        assert kernel_size in (3, 5)
        hidden = int(round(in_ch * expand_ratio))
        self.use_residual = stride == 1 and in_ch == out_ch
        pad = kernel_size // 2

        layers = []
        if expand_ratio != 1:
            layers += [
                nn.Conv2d(in_ch, hidden, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(hidden),
                nn.Hardswish(inplace=True),
            ]
        else:
            hidden = in_ch

        layers += [
            nn.Conv2d(hidden, hidden, kernel_size=kernel_size, stride=stride, padding=pad, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.Hardswish(inplace=True),
        ]
        self.expand_dw = nn.Sequential(*layers)
        self.se = SqueezeExciteMobileNetV3(hidden, se_ratio=se_ratio) if use_se else nn.Identity()
        self.project = nn.Sequential(
            nn.Conv2d(hidden, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        y = self.expand_dw(x)
        y = self.se(y)
        y = self.project(y)
        if self.use_residual:
            return x + y
        return y


def _student_resnet_mobilevit2_ds_conv_factory(ds_conv_block_version: int):
    """Return ``ds_conv_block`` callable used by StudentResNetMobileViT2Net-style models."""

    def ds_conv_block_v1(in_channels, out_channels, stride=1):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def ds_conv_block_v2(in_channels, out_channels, stride=1, expand_ratio=2):
        return InvertedResidualMobileNetV2(
            in_channels, out_channels, stride=stride, expand_ratio=expand_ratio
        )

    def ds_conv_block_v3(
        in_channels,
        out_channels,
        stride=1,
        expand_ratio=2,
        use_se=True,
        se_ratio=4,
        kernel_size=3,
    ):
        return InvertedResidualMobileNetV3(
            in_channels,
            out_channels,
            stride=stride,
            expand_ratio=expand_ratio,
            use_se=use_se,
            se_ratio=se_ratio,
            kernel_size=kernel_size,
        )

    if ds_conv_block_version == 1:
        return ds_conv_block_v1
    if ds_conv_block_version == 2:
        return ds_conv_block_v2
    if ds_conv_block_version == 3:
        return ds_conv_block_v3
    raise ValueError(f"Invalid ds_conv_block_version: {ds_conv_block_version}")


def _build_ds_body_from_channel_schedule(
    ds_conv_block,
    channel_schedule: Tuple[int, ...],
) -> nn.Sequential:
    """
    ``len(channel_schedule) - 1`` DS blocks; block ``i`` maps ``schedule[i] → schedule[i+1]``.

    Stride is fixed at **2** for every block (always downsample at each stage transition),
    regardless of whether adjacent channel widths are equal.
    """
    if len(channel_schedule) < 2:
        raise ValueError(f"channel_schedule needs at least 2 widths, got {channel_schedule!r}")
    layers: List[nn.Module] = []
    for i in range(len(channel_schedule) - 1):
        c_in = channel_schedule[i]
        c_out = channel_schedule[i + 1]
        layers.append(ds_conv_block(c_in, c_out, stride=2))
    return nn.Sequential(*layers)



def _make_resnet_like_stem(
    in_channels: int,
    stem_out_channels: int,
    pretrained: bool,
    ) -> nn.Sequential:
    """
    ResNet-18 stem geometry: 7×7 conv stride 2, BN, ReLU, 3×3 max-pool stride 2.
    Output channel width is ``stem_out_channels`` (no separate 64→C ``stem_adapter``).

    When ``pretrained`` is True, weights are taken from ImageNet ResNet-18 ``conv1``/``bn1``
    by slicing the first ``stem_out_channels`` filters (requires ``stem_out_channels <= 64``).
    ``in_channels`` may differ from 3 via the same channel-mean init as
    ``_adapt_resnet18_conv1_for_channels``.
    """
    if stem_out_channels < 1:
        raise ValueError(f"stem_out_channels must be >= 1, got {stem_out_channels}")
    conv = nn.Conv2d(
        in_channels,
        stem_out_channels,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False,
    )
    bn = nn.BatchNorm2d(stem_out_channels)
    if pretrained:
        if stem_out_channels > 64:
            raise ValueError(
                "pretrained ResNet stem slicing supports stem_out_channels <= 64; "
                f"got {stem_out_channels}"
            )
        try:
            from torchvision.models import ResNet18_Weights

            rnet = tvm.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        except Exception:
            rnet = tvm.resnet18(pretrained=True)
        _adapt_resnet18_conv1_for_channels(rnet, in_channels, pretrained=True)
        with torch.no_grad():
            conv.weight.copy_(rnet.conv1.weight[:stem_out_channels])
            bn.weight.copy_(rnet.bn1.weight[:stem_out_channels])
            bn.bias.copy_(rnet.bn1.bias[:stem_out_channels])
            bn.running_mean.copy_(rnet.bn1.running_mean[:stem_out_channels])
            bn.running_var.copy_(rnet.bn1.running_var[:stem_out_channels])
    else:
        nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")
        nn.init.ones_(bn.weight)
        nn.init.zeros_(bn.bias)
    return nn.Sequential(
        conv,
        bn,
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
    )



class ResNetLikeStemFourStageDsBody(nn.Module):
    """
    **Custom-width ResNet-like stem** (no 64-channel conv1 and no ``stem_adapter``) plus a DS body
    built from ``channel_schedule``: ``len(schedule)-1`` blocks, ``schedule[0]`` must equal
    ``stem_width``. Stride is 1 for equal in/out widths, else 2 (see ``_build_ds_body_from_channel_schedule``).

    ``pretrained`` initializes the stem from ImageNet ResNet-18 (sliced filters); the DS body is
    randomly initialized.
    """

    def __init__(
        self,
        in_channels: int,
        stem_width: int,
        pretrained: bool,
        ds_conv_block,
        channel_schedule: Sequence[int],
    ):
        super().__init__()
        sch = tuple(int(x) for x in channel_schedule)
        if len(sch) < 2:
            raise ValueError(f"channel_schedule needs at least 2 entries, got {sch!r}")
        if sch[0] != stem_width:
            raise ValueError(
                f"channel_schedule[0] ({sch[0]}) must equal stem_width ({stem_width})"
            )
        self.pretrained_requested = bool(pretrained)
        self.stem = _make_resnet_like_stem(in_channels, stem_width, pretrained=pretrained)
        self.channel_schedule = sch
        self.body = _build_ds_body_from_channel_schedule(ds_conv_block, sch)
        self.out_channels = sch[-1]
        self._init_weights()

    def _init_weights(self):
        for m in self.body.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        if not self.pretrained_requested:
            for m in self.stem.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        return self.body(x)


class DirectStem16Image_v2(nn.Module):
    """
    Image-only variant with the same temporal head style.
    """

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
        self.name = "DirectStem16Image_v2"
        ds_conv_block = _student_resnet_mobilevit2_ds_conv_factory(ds_conv_block_version)

        stem_w = 16
        channel_schedule = (16, 32, 64, 128, 256)

        self.image_encoder = ResNetLikeStemFourStageDsBody(
            image_channels,
            stem_w,
            resnet_pretrained,
            ds_conv_block,
            channel_schedule,
        )
        # self.image_global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.image_global_max_pool = nn.AdaptiveMaxPool2d(1)

        fused_dim = self.image_encoder.out_channels
        self.fusion_layer = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, feature_size),
        )

        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=gru_hidden_size,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True,
        )
        self.GRU = nn.GRU(
            input_size=gru_input_size,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            dropout=0.5 if gru_num_layers > 1 else 0,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(gru_input_size)
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def _image_encode(self, img):
        x = self.image_encoder.stem(img)
        for ds_block in self.image_encoder.body:
            x = ds_block(x)
        return x

    def forward(self, image_batch, radar_batch=None, beam=None):
        b, t, c, h, w = image_batch.shape
        img = image_batch.reshape(b * t, c, h, w)

        img_feat = self._image_encode(img)

        # img_avg = self.image_global_avg_pool(img_feat).flatten(1)
        img_max = self.image_global_max_pool(img_feat).flatten(1)
        # img_pooled = torch.cat([img_avg, img_max], dim=1)

        fused_features = self.fusion_layer(img_max).view(b, t, -1)

        features = self.layer_norm(fused_features)
        seq_out, _ = self.GRU(features)
        attn_output, _ = self.multihead_attention(seq_out, seq_out, seq_out)
        enhanced_seq_out = attn_output + seq_out

        pred = self.classifier(enhanced_seq_out)
        return pred, features, enhanced_seq_out