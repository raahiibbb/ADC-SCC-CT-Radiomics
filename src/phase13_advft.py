"""Phase 13c - site-adversarial fine-tuning of the FMCIB foundation model.

Pre-registered in docs/PHASE13_PREREGISTRATION.md (13c addendum).

  cache  : 64 mm crops around the MedSAM2 centroid -> 8 views (view 0 plain,
           1-7 augmented: shift, flips, in-plane rot90, HU shift/noise, blur)
           -> frozen FMCIB conv1..layer4[1] -> (N, 8, 4096, 2, 2, 2) fp16 memmap
  pooled : 5 seeds x 5 folds (same outer splits as phase11_cv), train the last
           bottleneck block + heads; writes results/phase13/pooled/<name>__combat
  loso   : leave-one-site-out, 5 seeds; writes results/phase13/loso/<name>__combat
  (the "__combat" suffix is only a directory convention so that the Phase-11/12
   fusion code finds the block; NO ComBat is applied - the adversary replaces it)

Variants: ft (lambda 0) and advft (label-conditional gradient-reversal site
adversary, DANN ramp to lambda_max 0.3).  No TCGA data is touched.

Usage (.venv-phase10):
  python src/phase13_advft.py cache
  python src/phase13_advft.py pooled --name advft --lam 0.3
  python src/phase13_advft.py loso   --name advft --lam 0.3
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import zlib

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torch.nn as nn
from scipy import ndimage as ndi
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import phase10_extract_fmcib as F  # noqa: E402
from phase10_cv import sb_auc, sb_weights, site_aucs  # noqa: E402

MASKS = "C:/LUNG_phase13/medsam2_masks"
CACHE = "C:/LUNG_phase13/advft_cache"
RES = os.path.join(ROOT, "results", "phase13")
COH = os.path.join(ROOT, "cohort", "phase12_four_site_cohort.csv")
SITES = ("LUNG1", "RADIOGENOMICS", "LPCD", "NLST")
SEEDS = [42, 43, 44, 45, 46]
K, BIG = 8, 64
HP = dict(lr_block=1e-4, lr_head=1e-3, wd=1e-2, batch=32, max_epochs=25, patience=6, dropout=0.3)
DEV = "cuda"


# ------------------------------------------------------------------ cache
def crop64(ctp, key):
    ct = sitk.ReadImage(ctp, sitk.sitkFloat32)
    st = sitk.LabelShapeStatisticsImageFilter()
    st.Execute(sitk.ReadImage(os.path.join(MASKS, key.replace("::", "__"), "AUTO_GTV.nii.gz"), sitk.sitkUInt8) > 0)
    seed = np.array(st.GetCentroid(1))
    ref = sitk.Image([BIG] * 3, sitk.sitkFloat32)
    ref.SetSpacing((1.0, 1.0, 1.0))
    ref.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
    ref.SetOrigin(tuple(seed - BIG // 2))
    return sitk.GetArrayFromImage(sitk.Resample(ct, ref, sitk.Transform(), sitk.sitkLinear, -1024.0, sitk.sitkFloat32))


def views(big_hu, key):
    rng = np.random.RandomState(zlib.crc32(key.encode()) & 0x7FFFFFFF)
    c, s, out = (BIG - F.SIZE) // 2, F.SIZE, []
    for v in range(K):
        if v == 0:
            x = big_hu[c:c + s, c:c + s, c:c + s].copy()
        else:
            o = c + rng.randint(-6, 7, size=3)
            x = big_hu[o[0]:o[0] + s, o[1]:o[1] + s, o[2]:o[2] + s].copy()
            if rng.rand() < 0.5:
                x = x[:, :, ::-1]
            if rng.rand() < 0.5:
                x = x[:, ::-1, :]
            x = np.rot90(x, rng.randint(4), axes=(1, 2))
            if rng.rand() < 0.5:
                x = ndi.gaussian_filter(x, sigma=rng.uniform(0.3, 1.0))
            x = x + rng.uniform(-30, 30) + rng.normal(0, rng.uniform(0, 20), x.shape)
        out.append((np.ascontiguousarray(x) + 1024.0) / 3072.0)
    return np.stack(out).astype(np.float32)


def frozen_trunk(m):
    return nn.Sequential(m.conv1, m.bn1, m.act, m.maxpool, m.layer1, m.layer2, m.layer3, m.layer4[0], m.layer4[1])


def cache():
    coh = pd.read_csv(COH)
    os.makedirs(CACHE, exist_ok=True)
    fn = os.path.join(CACHE, "l4b1.fp16")
    done_fn = os.path.join(CACHE, "done.npy")
    N = len(coh)
    mm = np.memmap(fn, dtype=np.float16, mode="r+" if os.path.isfile(fn) else "w+", shape=(N, K, 4096, 2, 2, 2))
    done = np.load(done_fn) if os.path.isfile(done_fn) else np.zeros(N, bool)
    trunk = frozen_trunk(F.load_model(DEV)).eval()
    t0 = time.time()
    for i, (k, ctp) in enumerate(zip(coh.PatientKey, coh.ct_path)):
        if done[i]:
            continue
        x = torch.from_numpy(views(crop64(ctp, k), k)[:, None]).to(DEV)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            mm[i] = trunk(x).float().cpu().numpy().astype(np.float16)
        done[i] = True
        if (i + 1) % 50 == 0:
            mm.flush()
            np.save(done_fn, done)
            print(f"{i + 1}/{N} {time.time() - t0:.0f}s", flush=True)
    mm.flush()
    np.save(done_fn, done)
    json.dump(dict(keys=list(coh.PatientKey), K=K, shape=[N, K, 4096, 2, 2, 2], masks=MASKS),
              open(os.path.join(CACHE, "index.json"), "w"))
    print("cache done", int(done.sum()), "/", N)


# ------------------------------------------------------------------ model
class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


class Net(nn.Module):
    def __init__(self, block, n_sites, p):
        super().__init__()
        self.block = block
        self.head = nn.Sequential(nn.Dropout(p), nn.Linear(4096, 1))
        self.adv = nn.Sequential(nn.Linear(4096 + 2, 64), nn.ReLU(), nn.Linear(64, n_sites))

    def forward(self, x, y=None, lam=0.0):
        f = self.block(x).mean((2, 3, 4))
        logit = self.head(f).squeeze(-1)
        if y is None:
            return logit, f
        yo = nn.functional.one_hot(y.long(), 2).float()
        return logit, self.adv(torch.cat([GRL.apply(f, lam), yo], 1))


def bn_eval(m):
    for mod in m.modules():
        if isinstance(mod, nn.BatchNorm3d):
            mod.eval()


def predict(net, Xc, idx):
    net.eval()
    out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for i in range(0, len(idx), 16):
            b = torch.from_numpy(Xc[idx[i:i + 16]].astype(np.float32)).to(DEV)     # (b, K, 4096, 2,2,2)
            bb = b.reshape(-1, *b.shape[2:])
            lo, _ = net(bb)
            out.append(lo.float().reshape(b.shape[0], K).mean(1).cpu().numpy())
    return np.concatenate(out)


def train(Xc, y, site, tr, seed, lam_max, base_block):
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    strata = np.array([f"{a}|{b}" for a, b in zip(site[tr], y[tr])])
    a, v = train_test_split(tr, test_size=0.2, stratify=strata, random_state=seed)
    s_names = sorted(set(site[tr]))
    s_idx = {s: i for i, s in enumerate(s_names)}
    net = Net(copy.deepcopy(base_block), len(s_names), HP["dropout"]).to(DEV)
    opt = torch.optim.AdamW([{"params": net.block.parameters(), "lr": HP["lr_block"]},
                             {"params": list(net.head.parameters()) + list(net.adv.parameters()), "lr": HP["lr_head"]}],
                            weight_decay=HP["wd"])
    w_all = sb_weights(site[a], y[a])
    w_all = w_all / w_all.mean()
    scaler = torch.amp.GradScaler("cuda")
    best, best_state, bad, best_ep = -1.0, None, 0, 0
    for ep in range(HP["max_epochs"]):
        p = ep / HP["max_epochs"]
        lam = lam_max * (2.0 / (1.0 + np.exp(-10 * p)) - 1.0)
        net.train()
        bn_eval(net)
        perm = rng.permutation(len(a))
        for i in range(0, len(a), HP["batch"]):
            j = perm[i:i + HP["batch"]]
            ids = a[j]
            vv = rng.randint(K, size=len(ids))
            xb = torch.from_numpy(np.stack([Xc[q, r] for q, r in zip(ids, vv)]).astype(np.float32)).to(DEV)
            yb = torch.tensor(y[ids], dtype=torch.float32, device=DEV)
            sb = torch.tensor([s_idx[s] for s in site[ids]], device=DEV)
            wb = torch.tensor(w_all[j], dtype=torch.float32, device=DEV)
            with torch.autocast("cuda", dtype=torch.float16):
                logit, slog = net(xb, yb, lam)
                l_lab = (nn.functional.binary_cross_entropy_with_logits(logit.float(), yb, reduction="none") * wb).mean()
                l_site = (nn.functional.cross_entropy(slog.float(), sb, reduction="none") * wb).mean()
                loss = l_lab + (l_site if lam_max > 0 else 0.0 * l_site)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sv = predict(net, Xc, v)
        auc = sb_auc(y[v], sv, site[v]) if len(np.unique(site[v])) > 1 else roc_auc_score(y[v], sv)
        if auc > best:
            best, best_ep, bad = auc, ep, 0
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= HP["patience"]:
                break
    net.load_state_dict(best_state)
    return net, best_ep, best


def site_leak(net, Xc, y, site, tr, te):
    """label-balanced one-vs-rest site AUC of a logistic probe on the learned
    features (fit on train, scored on test) - lower = more site-invariant."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    def feats(idx):
        net.eval()
        out = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            for i in range(0, len(idx), 16):
                b = torch.from_numpy(Xc[idx[i:i + 16], 0].astype(np.float32)).to(DEV)
                out.append(net(b)[1].float().cpu().numpy())
        return np.concatenate(out)
    ftr, fte = feats(tr), feats(te)
    sc = StandardScaler().fit(ftr)
    ftr, fte = sc.transform(ftr), sc.transform(fte)
    res = []
    for s in np.unique(site[tr]):
        t_tr, t_te = (site[tr] == s).astype(int), (site[te] == s).astype(int)
        if t_te.min() == t_te.max():
            continue
        m = LogisticRegression(C=0.1, max_iter=3000).fit(ftr, t_tr, sample_weight=sb_weights(site[tr], y[tr]))
        res.append(roc_auc_score(t_te, m.decision_function(fte), sample_weight=sb_weights(site[te], y[te])))
    return float(np.mean(res))


def load():
    coh = pd.read_csv(COH)
    idx = json.load(open(os.path.join(CACHE, "index.json")))
    assert idx["keys"] == list(coh.PatientKey)
    Xc = np.memmap(os.path.join(CACHE, "l4b1.fp16"), dtype=np.float16, mode="r", shape=tuple(idx["shape"]))
    Xc = np.asarray(Xc)                         # ~0.55 GB in RAM
    base = F.load_model("cpu").layer4[2]
    return coh, Xc, base


def run_pooled(a):
    coh, Xc, base = load()
    y, site = coh.label.values.astype(int), coh.site.values
    strata = np.array([f"{s}|{l}" for s, l in zip(site, y)])
    rows, meta, t0 = [], [], time.time()
    for seed in SEEDS:
        for f, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(Xc[:, 0, 0, 0, 0, 0], strata)):
            net, ep, inner = train(Xc, y, site, tr, seed + 1000 * f, a.lam, base)
            s = predict(net, Xc, te)
            leak = site_leak(net, Xc, y, site, tr, te)
            rows.append(pd.DataFrame(dict(idx=te, seed=seed, fold=f, ensemble=s)))
            meta.append(dict(seed=seed, fold=f, best_epoch=ep, inner_sb_auc=inner, site_leak_auc=leak))
            print(f"seed {seed} fold {f} ep {ep} inner {inner:.3f} leak {leak:.3f} {time.time() - t0:.0f}s", flush=True)
    df = pd.concat(rows, ignore_index=True)
    df.insert(1, "PatientKey", coh.PatientKey.values[df.idx])
    df.insert(2, "site", site[df.idx])
    df.insert(3, "y", y[df.idx])
    d = os.path.join(RES, "pooled", f"{a.name}__combat")
    os.makedirs(d, exist_ok=True)
    df.to_csv(os.path.join(d, "oof.csv"), index=False)
    per = pd.DataFrame([dict(sb_auc=sb_auc(g.y.values, g.ensemble.values, g.site.values),
                             **site_aucs(g.y.values, g.ensemble.values, g.site.values))
                        for _, g in df.groupby("seed")])
    met = dict(block=a.name, lam_max=a.lam, hp=HP, K=K, ensemble_mean=per.mean().to_dict(),
               ensemble_sd=per.std(ddof=1).to_dict(),
               site_leak_auc=float(np.mean([m["site_leak_auc"] for m in meta])), folds=meta,
               seconds=round(time.time() - t0, 1), note="no ComBat; directory suffix is a naming convention")
    json.dump(met, open(os.path.join(d, "metrics.json"), "w"), indent=1)
    e = met["ensemble_mean"]
    print(f"{a.name:8s} sbAUC {e['sb_auc']:.3f} | " + " ".join(f"{s} {e.get(s, float('nan')):.3f}" for s in SITES)
          + f" | siteLeak {met['site_leak_auc']:.3f}")


def run_loso(a):
    coh, Xc, base = load()
    y, site = coh.label.values.astype(int), coh.site.values
    rows = []
    for held in SITES:
        tr, te = np.where(site != held)[0], np.where(site == held)[0]
        for seed in SEEDS:
            net, ep, inner = train(Xc, y, site, tr, seed, a.lam, base)
            rows.append(pd.DataFrame(dict(idx=te, seed=seed, held_out=held, score=predict(net, Xc, te))))
            print(f"held {held} seed {seed} ep {ep} inner {inner:.3f}", flush=True)
    df = pd.concat(rows, ignore_index=True)
    df["y"] = y[df.idx]
    df["PatientKey"] = coh.PatientKey.values[df.idx]
    d = os.path.join(RES, "loso", f"{a.name}__combat")
    os.makedirs(d, exist_ok=True)
    df.to_csv(os.path.join(d, "scores.csv"), index=False)
    aucs = df.groupby(["held_out", "seed"]).apply(lambda g: roc_auc_score(g.y, g.score)).groupby(level=0).mean()
    print(f"{a.name:8s} LOSO " + " ".join(f"{k} {v:.3f}" for k, v in aucs.items()) + f" | mean {aucs.mean():.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["cache", "pooled", "loso"])
    ap.add_argument("--name", default="advft")
    ap.add_argument("--lam", type=float, default=0.3)
    a = ap.parse_args()
    {"cache": lambda: cache(), "pooled": lambda: run_pooled(a), "loso": lambda: run_loso(a)}[a.step]()


if __name__ == "__main__":
    main()
