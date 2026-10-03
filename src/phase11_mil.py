"""Phase 11 - gated attention-MIL over multi-crop FMCIB bags.

Instances = 16 frozen FMCIB embeddings per patient (centroid + boundary
crops).  Model: LayerNorm -> Linear(4096, 64) -> ReLU -> Dropout -> gated
attention (Ilse et al. 2018) -> linear head.  Site x label balanced BCE.
ComBat is applied to the INSTANCE embeddings, fitted on training patients only.
Epoch count chosen by an inner validation split (training patients only).

Same outer folds as phase11_cv (pooled) and the same leave-one-site-out
design (loso).  Outputs match phase11_cv so phase11_cv.py fusion can use them
as block "mil".

Usage: python src/phase11_mil.py pooled | loso
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase10_cv import sb_auc, sb_weights, site_aucs  # noqa: E402
from phase10_harmonise import Harmoniser  # noqa: E402
from phase11_cv import RES, SITES, cohort  # noqa: E402
from phase11_features import OUT  # noqa: E402

SEEDS = [42, 43, 44, 45, 46]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
HP = dict(hidden=64, att=32, dropout=0.3, lr=5e-4, wd=1e-3, max_epochs=60, patience=10, pca=None)


class GatedMIL(nn.Module):
    def __init__(self, d, h, a, p):
        super().__init__()
        self.enc = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, h), nn.ReLU(), nn.Dropout(p))
        self.V, self.U, self.w = nn.Linear(h, a), nn.Linear(h, a), nn.Linear(a, 1)
        self.head = nn.Linear(h, 1)

    def forward(self, x):                      # x: (B, K, d)
        h = self.enc(x)
        a = self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))).squeeze(-1)
        a = torch.softmax(a, dim=1)
        z = (a.unsqueeze(-1) * h).sum(1)
        return self.head(z).squeeze(-1), a


def load_bags(coh):
    return np.stack([np.load(os.path.join(OUT, "fmcib_bags", k.replace("::", "__") + ".npy")).astype(np.float32)
                     for k in coh.PatientKey])      # (N, K, 4096)


def harmonise(B, site, y, tr, te_sites=None):
    """ComBat on instances; each instance inherits its patient's site/label."""
    N, K, D = B.shape
    h = Harmoniser("combat").fit(B[tr].reshape(-1, D), np.repeat(site[tr], K), np.repeat(y[tr], K))
    out = np.empty_like(B)
    known = np.isin(site, list(h.gamma_))
    out[known] = h.transform(B[known].reshape(-1, D), np.repeat(site[known], K)).reshape(-1, K, D)
    for s in np.unique(site[~known]):          # unseen site: label-free location/scale
        m = site == s
        h.fit_new_site(B[m].reshape(-1, D), s, balanced=False)
        out[m] = h.transform(B[m].reshape(-1, D), np.repeat(site[m], K)).reshape(-1, K, D)
    mu = out[tr].reshape(-1, D).mean(0)
    sd = out[tr].reshape(-1, D).std(0) + 1e-6
    return (out - mu) / sd


def train(B, y, site, tr, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    strata = np.array([f"{a}|{b}" for a, b in zip(site[tr], y[tr])])
    a, v = train_test_split(tr, test_size=0.2, stratify=strata, random_state=seed)
    w = torch.tensor(sb_weights(site[a], y[a]), dtype=torch.float32, device=DEV)
    Xa = torch.tensor(B[a], device=DEV)
    ya = torch.tensor(y[a], dtype=torch.float32, device=DEV)
    Xv = torch.tensor(B[v], device=DEV)
    m = GatedMIL(B.shape[2], HP["hidden"], HP["att"], HP["dropout"]).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=HP["lr"], weight_decay=HP["wd"])
    best, best_ep, bad, state = -1, 0, 0, None
    for ep in range(HP["max_epochs"]):
        m.train()
        perm = torch.randperm(len(a), device=DEV)
        for i in range(0, len(a), 32):
            idx = perm[i:i + 32]
            logit, _ = m(Xa[idx])
            loss = (nn.functional.binary_cross_entropy_with_logits(logit, ya[idx], reduction="none") * w[idx]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            sv = m(Xv)[0].cpu().numpy()
        auc = sb_auc(y[v], sv, site[v]) if len(np.unique(site[v])) > 1 else roc_auc_score(y[v], sv)
        if auc > best:
            best, best_ep, bad = auc, ep, 0
            state = {k: t.detach().clone() for k, t in m.state_dict().items()}
        else:
            bad += 1
            if bad >= HP["patience"]:
                break
    m.load_state_dict(state)
    m.eval()
    return m, best_ep


def predict(m, B, idx):
    with torch.no_grad():
        s, a = m(torch.tensor(B[idx], device=DEV))
    return s.cpu().numpy(), a.cpu().numpy()


def run_pooled(coh, B):
    y, site = coh.y.values, coh.site.values
    strata = np.array([f"{a}|{b}" for a, b in zip(site, y)])
    rows = []
    for seed in SEEDS:
        for f, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(B, strata)):
            H = harmonise(B, site, y, tr)
            m, ep = train(H, y, site, tr, seed + f)
            s, att = predict(m, H, te)
            rows.append(pd.DataFrame(dict(idx=te, seed=seed, fold=f, ensemble=s, epoch=ep,
                                          att_max=att.max(1), att_centroid=att[:, 0])))
        print("seed", seed, "done", flush=True)
    df = pd.concat(rows, ignore_index=True)
    df.insert(1, "PatientKey", coh.PatientKey.values[df.idx])
    df.insert(2, "site", site[df.idx])
    df.insert(3, "y", y[df.idx])
    d = os.path.join(RES, "pooled", "mil__combat")
    os.makedirs(d, exist_ok=True)
    df.to_csv(os.path.join(d, "oof.csv"), index=False)
    per = pd.DataFrame([dict(sb_auc=sb_auc(g.y.values, g.ensemble.values, g.site.values),
                             **site_aucs(g.y.values, g.ensemble.values, g.site.values))
                        for _, g in df.groupby("seed")]).mean()
    print("MIL pooled", per.round(3).to_dict())


def run_loso(coh, B):
    y, site = coh.y.values, coh.site.values
    rows = []
    for held in SITES:
        tr, te = np.where(site != held)[0], np.where(site == held)[0]
        for seed in SEEDS:
            H = harmonise(B, site, y, tr)
            m, _ = train(H, y, site, tr, seed)
            s, _ = predict(m, H, te)
            rows.append(pd.DataFrame(dict(idx=te, seed=seed, held_out=held, score=s)))
    df = pd.concat(rows, ignore_index=True)
    df["y"] = y[df.idx]
    df["PatientKey"] = coh.PatientKey.values[df.idx]
    d = os.path.join(RES, "loso", "mil__combat")
    os.makedirs(d, exist_ok=True)
    df.to_csv(os.path.join(d, "scores.csv"), index=False)
    aucs = df.groupby(["held_out", "seed"]).apply(lambda g: roc_auc_score(g.y, g.score)).groupby(level=0).mean()
    print("MIL LOSO", aucs.round(3).to_dict(), "mean %.3f" % aucs.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["pooled", "loso"])
    a = ap.parse_args()
    coh = cohort()
    B = load_bags(coh)
    print("bags", B.shape, "device", DEV, flush=True)
    run_pooled(coh, B) if a.step == "pooled" else run_loso(coh, B)


if __name__ == "__main__":
    main()
