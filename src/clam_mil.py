"""Phase 6C - CLAM-inspired instance-level supervision on top of the FROZEN
Phase-6B gated-attention MIL model.

CLAM (Lu et al. 2021) is a REFERENCE IMPLEMENTATION ONLY.  Nothing under
`references/CLAM-master/` is imported, vendored, adapted line-by-line or
executed.  What is borrowed here is one CONCEPT:

    inside a TRAINING bag, the instances the model currently attends to most are
    treated as pseudo-positive evidence and the ones it attends to least as
    pseudo-negative evidence, and a small auxiliary classifier is trained to tell
    those two groups apart.

Three things this is NOT, stated once and enforced by the audit:

  * The pseudo targets are **model-generated instance-clustering targets**.  They
    are NOT histologic ADC/SCC patch labels and must never be described as such.
    The patient label never reaches an individual instance.
  * No separate patch-level ADC-vs-SCC classifier exists.  The auxiliary task
    only asks whether an instance belongs to the model's own high-attention or
    low-attention group of that same bag.
  * Nothing is precomputed.  The top-k / bottom-k sets are recomputed from the
    CURRENT model's attention on every forward pass of every epoch.

Lu's fixed B = 8 is deliberately NOT copied.  Our bags hold 8 to 900 instances,
so a fixed top-8 / bottom-8 would select the same 8 instances twice, with
opposite targets, in the smallest bag.  The adaptation is:

    k = min(8, max(1, floor(0.10 * N))),   and if 2*k > N,  k = floor(N / 2)

which gives k = 1 at N = 8 and N = 10, 2 at N = 20, 5 at N = 50 and 8 from
N = 80 upward, and guarantees 2*k <= N so the two sets are always disjoint.

The bag-level architecture is inherited from `mil_models.GatedAttentionMIL`
unchanged - same 512 -> 64 -> 32 encoder, same gated attention, same linear
patient classifier - so the ONLY substantive addition is the instance head and
the instance loss.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from mil_models import GatedAttentionMIL

MODEL_NAME = "clam_attention_mil"


# --------------------------------------------------------------------- adaptive k
def adaptive_k(n: int, fraction: float = 0.10, max_k: int = 8, min_k: int = 1) -> int:
    """Instances per pseudo-class for a bag of `n` instances.

        k = min(max_k, max(min_k, floor(fraction * n)))
        if 2*k > n:  k = floor(n / 2)          # deterministic reduction

    Returns 0 only for a degenerate bag (n < 2), in which case the caller must
    skip instance supervision for that bag entirely.  This project's smallest bag
    holds 8 instances, so 0 never occurs on real data.
    """
    n = int(n)
    if n < 2:
        return 0
    k = min(int(max_k), max(int(min_k), int(math.floor(float(fraction) * n))))
    if 2 * k > n:
        k = int(math.floor(n / 2))
    return int(max(k, 0))


def k_rule_from_config(cfg: dict) -> Dict[str, float]:
    r = cfg["instance_supervision"]["k_rule"]
    return {"fraction": float(r["fraction"]), "max_k": int(r["max_k"]),
            "min_k": int(r["min_k"])}


def bag_k(n: int, rule: Dict[str, float]) -> int:
    return adaptive_k(n, rule["fraction"], rule["max_k"], rule["min_k"])


# ------------------------------------------------------------------------ model
class ClamStyleGatedAttentionMIL(GatedAttentionMIL):
    """Phase-6B `GatedAttentionMIL` + one small binary linear instance head.

    Every bag-level tensor - `encoder.*`, `att_V`, `att_U`, `att_w`,
    `classifier` - is inherited unchanged, with the same names and the same
    shapes, so a checkpoint of this model differs from a Phase-6B attention
    checkpoint by exactly the two `instance_classifier` tensors.

    `forward` keeps the Phase-6B `(logit, alpha)` signature so the frozen
    `predict()` path can score this model without modification.  `encode()`
    additionally returns the 32-D encoded instance representation, which is what
    the instance head consumes - the SAME representation the attention module and
    the bag classifier see.
    """

    name = MODEL_NAME

    def __init__(self, input_dim: int, hidden: List[int], dropout: float,
                 attention_hidden: int = 32, **_ignored):
        super().__init__(input_dim, hidden, dropout, attention_hidden)
        self.instance_classifier = nn.Linear(self.encoder.out_dim, 1)

    # arithmetic identical to GatedAttentionMIL.forward, with `h` also returned
    def encode(self, bag: torch.Tensor):
        h = self.encoder(bag)                                                    # [N, 32]
        s = self.att_w(torch.tanh(self.att_V(h)) * torch.sigmoid(self.att_U(h)))  # [N, 1]
        alpha = torch.softmax(s, dim=0)
        z = (alpha * h).sum(dim=0, keepdim=True)                                  # [1, 32]
        logit = self.classifier(z).squeeze(-1).squeeze(0)
        return logit, alpha.squeeze(-1), h

    def forward(self, bag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logit, alpha, _ = self.encode(bag)
        return logit, alpha

    def bag_level_state_dict(self) -> Dict[str, torch.Tensor]:
        """The state a Phase-6B `GatedAttentionMIL` would carry - no instance head."""
        return {k: v for k, v in self.state_dict().items()
                if not k.startswith("instance_classifier.")}


def build_clam_model(cfg: dict, input_dim: int) -> ClamStyleGatedAttentionMIL:
    m = cfg["model"]
    return ClamStyleGatedAttentionMIL(
        input_dim=int(input_dim),
        hidden=list(m["encoder_hidden"]),
        dropout=float(m["dropout"]),
        attention_hidden=int(m["attention_hidden"]))


# ------------------------------------------------------- pseudo-instance selection
def select_pseudo_instances(alpha: torch.Tensor, k: int):
    """Top-k and bottom-k instance indices of the CURRENT attention vector.

    A stable descending sort makes the choice deterministic when weights tie, and
    `2*k <= N` (guaranteed by `adaptive_k`) makes the two index sets disjoint by
    construction.  Selection is an indexing operation: no gradient flows through
    the ranking itself, exactly as in CLAM's `inst_eval`.
    """
    n = int(alpha.shape[0])
    if k < 1 or 2 * k > n:
        raise ValueError("invalid k=%d for a bag of %d instances" % (k, n))
    order = torch.argsort(alpha.detach(), descending=True, stable=True)
    top = order[:k]
    bottom = order[n - k:]
    return top, bottom


def instance_loss_for_bag(model: ClamStyleGatedAttentionMIL, alpha: torch.Tensor,
                          h: torch.Tensor, k: int, loss_fn: nn.Module,
                          device: torch.device):
    """BCE of the auxiliary head on `k` pseudo-positive + `k` pseudo-negative rows.

    The two groups are equal in size within every bag, so the pseudo targets are
    balanced by construction and the ADC/SCC `pos_weight` must NOT be applied
    here.  Returns the loss and a per-bag diagnostic record.
    """
    top, bottom = select_pseudo_instances(alpha, k)
    idx = torch.cat([top, bottom], dim=0)
    logits = model.instance_classifier(h[idx]).squeeze(-1)              # [2k]
    targets = torch.cat([torch.ones(k, device=device),
                         torch.zeros(k, device=device)], dim=0)
    loss = loss_fn(logits, targets)

    with torch.no_grad():
        a = alpha.detach()
        pred = (torch.sigmoid(logits.detach()) >= 0.5).float()
        diag = {
            "k": int(k),
            "n_instances": int(alpha.shape[0]),
            "n_pseudo_instances": int(2 * k),
            "fraction_participating": float(2 * k) / float(alpha.shape[0]),
            "mean_top_attention": float(a[top].mean().item()),
            "mean_bottom_attention": float(a[bottom].mean().item()),
            "attention_separation": float((a[top].mean() - a[bottom].mean()).item()),
            "attention_separation_ratio": float(
                (a[top].mean() / torch.clamp(a[bottom].mean(), min=1e-30)).item()),
            "instance_pseudo_accuracy": float((pred == targets).float().mean().item()),
            "instance_loss": float(loss.detach().item()),
            "overlap": int(len(set(top.tolist()) & set(bottom.tolist()))),
        }
    return loss, diag


# ------------------------------------------------------------------ training step
def train_one_epoch_clam(model: ClamStyleGatedAttentionMIL, tensors: Dict[str, np.ndarray],
                         labels: Dict[str, int], order: Sequence[str], optimizer,
                         bag_loss_fn: nn.Module, inst_loss_fn: nn.Module,
                         device: torch.device, lambda_instance: float,
                         k_rule: Dict[str, float],
                         collect: Optional[List[dict]] = None) -> Dict[str, float]:
    """One epoch: one patient = one bag = one optimiser step.

        L_total = (1 - lambda_instance) * L_bag + lambda_instance * L_instance

    `L_bag` is the weighted BCE on the patient's ADC/SCC label; `L_instance` is
    the auxiliary BCE on that same bag's own attention ranking.  Pseudo-instances
    are never pooled across patients: the instance loss is computed inside the
    bag it came from and contributes to that bag's step only.
    """
    model.train()
    lam = float(lambda_instance)
    tot_bag = tot_inst = tot_total = 0.0
    n_bags = n_inst_bags = 0
    ks: List[int] = []
    acc_sum = sep_sum = frac_sum = 0.0

    for pid in order:
        optimizer.zero_grad()
        x = torch.from_numpy(tensors[pid]).to(device)
        logit, alpha, h = model.encode(x)
        target = torch.tensor([float(labels[pid])], dtype=torch.float32, device=device)
        l_bag = bag_loss_fn(logit.unsqueeze(0), target)

        k = bag_k(int(x.shape[0]), k_rule)
        if lam > 0.0 and k >= 1:
            l_inst, diag = instance_loss_for_bag(model, alpha, h, k, inst_loss_fn, device)
            n_inst_bags += 1
            ks.append(k)
            acc_sum += diag["instance_pseudo_accuracy"]
            sep_sum += diag["attention_separation"]
            frac_sum += diag["fraction_participating"]
            tot_inst += diag["instance_loss"]
            if collect is not None:
                diag = dict(diag)
                diag["PatientID"] = pid
                diag["bag_loss"] = float(l_bag.detach().item())
                collect.append(diag)
        else:
            l_inst = torch.zeros((), dtype=torch.float32, device=device)

        total = (1.0 - lam) * l_bag + lam * l_inst
        total.backward()
        optimizer.step()

        tot_bag += float(l_bag.detach().item())
        tot_total += float(total.detach().item())
        n_bags += 1

    m = max(n_bags, 1)
    mi = max(n_inst_bags, 1)
    return {
        "bag_loss": tot_bag / m,
        "instance_loss": (tot_inst / mi) if n_inst_bags else float("nan"),
        "total_loss": tot_total / m,
        "n_bags": n_bags,
        "n_bags_with_instance_loss": n_inst_bags,
        "mean_k": float(np.mean(ks)) if ks else float("nan"),
        "median_k": float(np.median(ks)) if ks else float("nan"),
        "min_k": int(np.min(ks)) if ks else 0,
        "max_k": int(np.max(ks)) if ks else 0,
        "mean_instance_pseudo_accuracy": (acc_sum / mi) if n_inst_bags else float("nan"),
        "mean_attention_separation": (sep_sum / mi) if n_inst_bags else float("nan"),
        "mean_fraction_participating": (frac_sum / mi) if n_inst_bags else float("nan"),
    }


@torch.no_grad()
def predict_clam(model: ClamStyleGatedAttentionMIL, tensors: Dict[str, np.ndarray],
                 pids: Sequence[str], device: torch.device,
                 want_attention: bool = False):
    """Outer-test / inner-validation inference.

    Bag probability and attention only.  **No instance loss is computed here and
    no parameter is updated**, so an outer-test patient never influences training.
    """
    model.eval()
    probs, attns = [], {}
    for pid in pids:
        logit, alpha = model(torch.from_numpy(tensors[pid]).to(device))
        probs.append(float(torch.sigmoid(logit).item()))
        if want_attention:
            attns[pid] = alpha.detach().cpu().numpy().astype(np.float64)
    return np.asarray(probs, dtype=float), attns
