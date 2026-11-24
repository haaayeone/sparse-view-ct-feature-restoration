import torch
import torch.nn as nn


class LinearProbe(nn.Module):
    """Single linear classifier on frozen embeddings (Stage 2)."""

    def __init__(self, in_dim: int, num_labels: int,
                 hidden: int = 512, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, num_labels),
        )

    def forward(self, x: torch.Tensor):
        logits = self.net(x.float())
        return [logits[:, i : i + 1] for i in range(logits.size(1))]
