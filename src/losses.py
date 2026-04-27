import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Args:
        pos_margin: Umbral de distancia para pares positivos (default=0)
        neg_margin: Umbral de distancia para pares negativos (default=1)
    """

    def __init__(self, pos_margin: float = 0.0, neg_margin: float = 1.0):
        super().__init__()
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:

        D = F.pairwise_distance(z1, z2) 

        # Positivos: penaliza si D > pos_margin
        pos_loss = torch.clamp(D - self.pos_margin, min=0.0) ** 2

        # Negativos: penaliza si D < neg_margin
        neg_loss = torch.clamp(self.neg_margin - D, min=0.0) ** 2

        loss = torch.mean(
            (1 - labels) * pos_loss +
            labels * neg_loss
        )
        return loss