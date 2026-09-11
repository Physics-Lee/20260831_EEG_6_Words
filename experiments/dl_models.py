"""Deep models for the 3-condition task, migrated from
D:\\repositories\\20260703_Preprocess\\code\\classifier_models.py (10-char ECoG)
and adapted: n_classes=3, 38 channels x 500 timepoints (500 Hz EEG).

Unified input convention: X = (B, C, T) — batch x channels x timepoints.
Output: logits (B, n_classes). Only GRU / LSTM / ViT are kept.
"""

from __future__ import annotations

import torch
import torch.nn as nn

N_CLASSES = 3


# ============================================================
# GRU — bidirectional GRU over the time axis.
# Input: (B, C, T) -> transpose to (B, T, C) -> GRU -> time mean-pool -> head.
# ============================================================
class GRUClassifier(nn.Module):
    def __init__(self, n_channels: int = 38, n_time: int = 500,
                 hidden: int = 128, n_layers: int = 2, dropout: float = 0.3,
                 n_classes: int = N_CLASSES, input_proj_dim: int | None = None):
        super().__init__()
        if input_proj_dim is not None and input_proj_dim > 0:
            self.input_proj = nn.Sequential(
                nn.Linear(n_channels, input_proj_dim),
                nn.Dropout(dropout),
            )
            gru_input_size = input_proj_dim
        else:
            self.input_proj = nn.Identity()
            gru_input_size = n_channels
        self.gru = nn.GRU(
            input_size=gru_input_size, hidden_size=hidden, num_layers=n_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, n_classes),
        )

    def forward(self, x):                  # x: (B, C, T)
        x = x.transpose(1, 2)              # -> (B, T, C)
        x = self.input_proj(x)
        out, _ = self.gru(x)               # (B, T, 2*hidden)
        pooled = out.mean(dim=1)           # (B, 2*hidden)
        return self.head(pooled)


# ============================================================
# LSTM — mirror of GRU, swapping nn.GRU for nn.LSTM.
# ============================================================
class LSTMClassifier(nn.Module):
    def __init__(self, n_channels: int = 38, n_time: int = 500,
                 hidden: int = 128, n_layers: int = 2, dropout: float = 0.3,
                 n_classes: int = N_CLASSES, input_proj_dim: int | None = None):
        super().__init__()
        if input_proj_dim is not None and input_proj_dim > 0:
            self.input_proj = nn.Sequential(
                nn.Linear(n_channels, input_proj_dim),
                nn.Dropout(dropout),
            )
            lstm_input_size = input_proj_dim
        else:
            self.input_proj = nn.Identity()
            lstm_input_size = n_channels
        self.lstm = nn.LSTM(
            input_size=lstm_input_size, hidden_size=hidden, num_layers=n_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, n_classes),
        )

    def forward(self, x):                  # x: (B, C, T)
        x = x.transpose(1, 2)              # -> (B, T, C)
        x = self.input_proj(x)
        out, _ = self.lstm(x)              # (B, T, 2*hidden)
        pooled = out.mean(dim=1)           # (B, 2*hidden)
        return self.head(pooled)


# ============================================================
# ViT — treats (C, T) as a 2D image, patchifies, standard Transformer encoder.
# (38, 500) -> patch (19, 50) -> 2 x 10 = 20 patches.
# ============================================================
class PatchEmbed(nn.Module):
    def __init__(self, img_h: int, img_w: int, patch: tuple[int, int],
                 in_ch: int = 1, dim: int = 128):
        super().__init__()
        assert img_h % patch[0] == 0 and img_w % patch[1] == 0, (
            f"image ({img_h},{img_w}) must be divisible by patch {patch}"
        )
        self.nh, self.nw = img_h // patch[0], img_w // patch[1]
        self.n_patches = self.nh * self.nw
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)

    def forward(self, x):                  # x: (B, in_ch, H, W)
        x = self.proj(x)                   # (B, dim, nh, nw)
        x = x.flatten(2).transpose(1, 2)   # (B, n_patches, dim)
        return x


class ViTClassifier(nn.Module):
    """Small ViT for a small per-subject dataset (~100-2000 samples). Default
    config heavily shrunk vs standard ViT: dim=128, depth=4, heads=4, mlp=2x."""

    def __init__(self, img_h: int = 38, img_w: int = 500,
                 patch: tuple[int, int] = (19, 50), in_ch: int = 1,
                 dim: int = 128, depth: int = 4, heads: int = 4,
                 mlp_ratio: float = 2.0, dropout: float = 0.3,
                 n_classes: int = N_CLASSES):
        super().__init__()
        self.patch = PatchEmbed(img_h, img_w, patch, in_ch, dim)
        n_patches = self.patch.n_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads,
            dim_feedforward=int(dim * mlp_ratio), dropout=dropout,
            batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, n_classes)

    def forward(self, x):                  # x: (B, C, T)
        x = x.unsqueeze(1)                 # (B, 1, C, T)
        x = self.patch(x)                  # (B, n_patches, dim)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)     # (B, n_patches+1, dim)
        x = x + self.pos_embed
        x = self.encoder(x)
        cls_out = self.norm(x[:, 0])
        return self.head(cls_out)


# ============================================================
# Factory
# ============================================================
def _pick_patch_h(n_channels: int, cap: int = 32) -> int:
    """Largest divisor of n_channels that is <= cap."""
    start = min(cap, n_channels)
    for h in range(start, 0, -1):
        if n_channels % h == 0:
            return h
    return 1


def build_model(name: str, n_channels: int = 38, n_time: int = 500,
                n_classes: int = N_CLASSES, device: str = "cpu", **kwargs):
    name = name.lower()
    if name == "gru":
        m = GRUClassifier(n_channels=n_channels, n_time=n_time,
                          n_classes=n_classes, **kwargs)
    elif name == "lstm":
        m = LSTMClassifier(n_channels=n_channels, n_time=n_time,
                           n_classes=n_classes, **kwargs)
    elif name == "vit":
        if "patch" not in kwargs:
            ph = _pick_patch_h(n_channels, cap=max(19, n_channels // 2))
            pw = _pick_patch_h(n_time, cap=50)
            kwargs["patch"] = (ph, pw)
        m = ViTClassifier(img_h=n_channels, img_w=n_time,
                          n_classes=n_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model: {name}")
    return m.to(device)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
