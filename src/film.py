"""
film.py
RO-2026-002 — ABPET Threshold
Frontier Translational Research Lab (Independent)
PI: Julian Borges, MD

Feature-wise Linear Modulation (FiLM) for tracer conditioning.
Reference: Perez et al. FiLM: Visual Reasoning with a General Conditioning Layer.
           AAAI 2018. arXiv:1709.07871

FiLM applies learned affine transformation per tracer at each encoder stage:
    y = gamma(tracer) * x + beta(tracer)
where x is the feature map tensor and gamma/beta are predicted from tracer embedding.
Applied after each residual stage BN, before ReLU.
"""

import torch
import torch.nn as nn


class FiLMLayer(nn.Module):
    """
    Single FiLM conditioning layer.

    Args:
        emb_dim:     Dimension of tracer embedding vector
        num_channels: Number of feature map channels to modulate

    Input:
        x:   (B, C, D, H, W) feature map
        emb: (B, emb_dim) tracer embedding

    Output:
        (B, C, D, H, W) modulated feature map
    """

    def __init__(self, emb_dim: int, num_channels: int):
        super().__init__()
        self.fc = nn.Linear(emb_dim, 2 * num_channels)
        self.num_channels = num_channels

        # Initialize to identity (gamma=1, beta=0) for stable training start
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        # Set gamma bias to 1
        with torch.no_grad():
            self.fc.bias[:num_channels] = 1.0

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        # emb: (B, emb_dim) -> params: (B, 2*C)
        params = self.fc(emb)
        gamma = params[:, :self.num_channels]   # (B, C)
        beta = params[:, self.num_channels:]    # (B, C)

        # Broadcast over spatial dims (D, H, W)
        gamma = gamma.view(-1, self.num_channels, 1, 1, 1)
        beta = beta.view(-1, self.num_channels, 1, 1, 1)

        return gamma * x + beta


class TracerEmbedding(nn.Module):
    """
    Tracer embedding lookup.

    Args:
        num_tracers: Number of distinct tracers (4: FBP, FBB, PIB, NAV)
        emb_dim:     Embedding dimension
    """

    def __init__(self, num_tracers: int = 4, emb_dim: int = 32):
        super().__init__()
        self.embedding = nn.Embedding(num_tracers, emb_dim)

    def forward(self, tracer_idx: torch.Tensor) -> torch.Tensor:
        return self.embedding(tracer_idx)  # (B, emb_dim)
