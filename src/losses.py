import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss from Hadsell et al. 2006.
    L = (1-Y) * 0.5 * D^2 + Y * 0.5 * max(0, margin - D)^2

    Y=0 → similar pair, Y=1 → dissimilar pair
    """
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, z1: torch.Tensor, z2: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        dist = F.pairwise_distance(z1, z2)  # (B,)

        pos_loss = (1 - labels) * 0.5 * dist ** 2
        neg_loss = labels * 0.5 * F.relu(self.margin - dist) ** 2

        return (pos_loss + neg_loss).mean()