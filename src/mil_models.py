"""Phase 3 - the two MIL models.

Both share an identical patch encoder (F -> 64 -> 32, ReLU, dropout 0.2) and an
identical linear patient classifier.  The pooling operator is the only
architectural difference:

    MeanMIL       z = (1/N) sum_i h_i
    AttentionMIL  z = sum_i alpha_i h_i,  alpha = softmax over the bag of the
                  gated attention scores  s_i = w^T [tanh(V h_i) * sigmoid(U h_i)]

One patient = one bag = one forward pass = one logit = one loss contribution.
Patches from different patients are never mixed inside a bag.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class PatchEncoder(nn.Module):
    """F -> 64 -> 32, ReLU, dropout 0.2 (applied after the first ReLU)."""

    def __init__(self, input_dim: int, hidden: List[int], dropout: float):
        super().__init__()
        h1, h2 = int(hidden[0]), int(hidden[1])
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(h1, h2),
            nn.ReLU(),
        )
        self.out_dim = h2

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # [N, F] -> [N, 32]
        return self.net(x)


class MeanMIL(nn.Module):
    name = "mean_mil"

    def __init__(self, input_dim: int, hidden: List[int], dropout: float, **_ignored):
        super().__init__()
        self.encoder = PatchEncoder(input_dim, hidden, dropout)
        self.classifier = nn.Linear(self.encoder.out_dim, 1)

    def forward(self, bag: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        h = self.encoder(bag)                     # [N, 32]
        z = h.mean(dim=0, keepdim=True)           # [1, 32]
        logit = self.classifier(z).squeeze(-1).squeeze(0)
        return logit, None


class GatedAttentionMIL(nn.Module):
    name = "attention_mil"

    def __init__(self, input_dim: int, hidden: List[int], dropout: float,
                 attention_hidden: int = 32, **_ignored):
        super().__init__()
        self.encoder = PatchEncoder(input_dim, hidden, dropout)
        d = self.encoder.out_dim
        a = int(attention_hidden)
        self.att_V = nn.Linear(d, a)
        self.att_U = nn.Linear(d, a)
        self.att_w = nn.Linear(a, 1)
        self.classifier = nn.Linear(d, 1)

    def forward(self, bag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(bag)                                    # [N, 32]
        s = self.att_w(torch.tanh(self.att_V(h)) * torch.sigmoid(self.att_U(h)))  # [N, 1]
        alpha = torch.softmax(s, dim=0)                           # normalised within the bag
        z = (alpha * h).sum(dim=0, keepdim=True)                  # [1, 32]
        logit = self.classifier(z).squeeze(-1).squeeze(0)
        return logit, alpha.squeeze(-1)


def build_model(model_name: str, cfg: dict, input_dim: Optional[int] = None) -> nn.Module:
    """`input_dim` overrides the configured 74 when a fold drops screened features."""
    mcfg = cfg["model"]
    kwargs = dict(input_dim=int(mcfg["input_dim"] if input_dim is None else input_dim),
                  hidden=list(mcfg["encoder_hidden"]),
                  dropout=float(mcfg["dropout"]),
                  attention_hidden=int(mcfg["attention_hidden"]))
    if model_name == "mean_mil":
        return MeanMIL(**kwargs)
    if model_name == "attention_mil":
        return GatedAttentionMIL(**kwargs)
    raise ValueError("unknown model %r" % model_name)
