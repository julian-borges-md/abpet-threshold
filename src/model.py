"""
model.py
RO-2026-002 — ABPET Threshold
Frontier Translational Research Lab (Independent)
PI: Julian Borges, MD

All six model variants for the ablation study.

Model  Architecture    Tracer Fusion   Pretrain
A      BaselineCNN     Concat 8-dim    No        (hackathon replica — anchor)
B      BaselineCNN     Concat 8-dim    No        (LR scheduler effect only)
C      BaselineCNN     Concat 8-dim    No        (+ augmentation)
D      ResNet18_3D     Concat 8-dim    No        (architecture effect)
E      ResNet18_3D     FiLM 32-dim     No        (tracer fusion effect)
F      ResNet18_3D     FiLM 32-dim     MedNet    (pretraining effect)
"""

import torch
import torch.nn as nn
from src.film import FiLMLayer, TracerEmbedding

NUM_TRACERS = 4


# ── BaselineCNN (exact hackathon architecture) ───────────────────────────────

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2)
        )

    def forward(self, x):
        return self.block(x)


class BaselineCNN(nn.Module):
    """
    Exact hackathon starter architecture.
    Used for Models A, B, C.
    Tracer: 8-dim embedding concatenated at head only.
    """

    def __init__(self, num_tracers=NUM_TRACERS, emb_dim=8):
        super().__init__()
        self.tracer_emb = nn.Embedding(num_tracers, emb_dim)
        self.encoder = nn.Sequential(
            ConvBlock(1, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(256 + emb_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x, tracer_idx):
        feat = self.encoder(x)
        feat = nn.AdaptiveAvgPool3d(1)(feat).flatten(1)
        emb = self.tracer_emb(tracer_idx)
        out = self.head[2:](torch.cat([feat, emb], dim=1))
        return out.squeeze(1)

# ── ResNet18 3D building blocks ───────────────────────────────────────────────

class ResBlock3D(nn.Module):
    """Standard ResNet18 residual block adapted for 3D."""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_ch)

        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm3d(out_ch)
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            identity = self.downsample(x)
        return self.relu(out + identity)


def make_layer(in_ch, out_ch, num_blocks, stride=2):
    layers = [ResBlock3D(in_ch, out_ch, stride=stride)]
    for _ in range(1, num_blocks):
        layers.append(ResBlock3D(out_ch, out_ch))
    return nn.Sequential(*layers)


# ── ResNet18_3D — Models D, E, F ─────────────────────────────────────────────

class ResNet18_3D(nn.Module):
    """
    ResNet18 adapted for 3D amyloid PET regression.
    Used for Models D (concat fusion), E (FiLM fusion), F (FiLM + MedNet init).

    Args:
        use_film:    If True, apply FiLM conditioning after each stage (Models E, F)
        emb_dim:     Tracer embedding dimension (8 for concat, 32 for FiLM)
        num_tracers: Number of tracer types
    """

    def __init__(self, use_film=False, emb_dim=32, num_tracers=NUM_TRACERS):
        super().__init__()
        self.use_film = use_film
        self.emb_dim = emb_dim

        self.tracer_emb = TracerEmbedding(num_tracers, emb_dim)

        # Stem
        self.stem = nn.Sequential(
            nn.Conv3d(1, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(3, stride=2, padding=1)
        )

        # Encoder stages [2, 2, 2, 2] blocks
        self.layer1 = make_layer(64, 64, 2, stride=1)
        self.layer2 = make_layer(64, 128, 2, stride=2)
        self.layer3 = make_layer(128, 256, 2, stride=2)
        self.layer4 = make_layer(256, 512, 2, stride=2)

        # FiLM layers (applied after each stage if use_film=True)
        if use_film:
            self.film1 = FiLMLayer(emb_dim, 64)
            self.film2 = FiLMLayer(emb_dim, 128)
            self.film3 = FiLMLayer(emb_dim, 256)
            self.film4 = FiLMLayer(emb_dim, 512)

        # Regression head
        head_in = 512 + (0 if use_film else emb_dim)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(head_in, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x, tracer_idx):
        emb = self.tracer_emb(tracer_idx)   # (B, emb_dim)
        x = self.stem(x)

        x = self.layer1(x)
        if self.use_film:
            x = self.film1(x, emb)

        x = self.layer2(x)
        if self.use_film:
            x = self.film2(x, emb)

        x = self.layer3(x)
        if self.use_film:
            x = self.film3(x, emb)

        x = self.layer4(x)
        if self.use_film:
            x = self.film4(x, emb)

        feat = nn.AdaptiveAvgPool3d(1)(x).flatten(1)

        if not self.use_film:
            feat = torch.cat([feat, emb], dim=1)

        return self.head[2:](feat).squeeze(1)


# ── Model factory ─────────────────────────────────────────────────────────────

def get_model(model_name: str) -> nn.Module:
    """
    Return the correct model instance by name.

    Args:
        model_name: One of 'A', 'B', 'C', 'D', 'E', 'F'

    Note: Models B and C use identical architecture to A.
    Training script applies the LR scheduler and augmentation differences.
    Model F uses MedicalNet pretrained weights — loaded separately in train.py.
    """
    name = model_name.upper()
    if name in ("A", "B", "C"):
        return BaselineCNN(emb_dim=8)
    elif name == "D":
        return ResNet18_3D(use_film=False, emb_dim=8)
    elif name in ("E", "F"):
        return ResNet18_3D(use_film=True, emb_dim=32)
    else:
        raise ValueError(f"Unknown model name '{model_name}'. Must be A-F.")
