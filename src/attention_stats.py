"""Phase 3B / Phase 5 - numerical concentration statistics of the saved attention weights.

Purely descriptive: how far each bag's attention distribution is from uniform.
It reads only the saved weights and produces numbers, never a map, an overlay or
any spatial/biological statement - that remains Phase-4 work.

    python src/attention_stats.py                          # Phase 3 + 3B Gradient
    python src/attention_stats.py --config config/phase5.yaml   # Phase 5 GTV+Rim
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mil_data import load_phase3_config, ppath  # noqa: E402

MODELS = ("mean_mil", "attention_mil")
DEFAULT_CONFIG = "config/phase3b.yaml"
# tag -> attention directory.  Overridable per phase via
# outputs.attention_sources in the config.
DEFAULT_SOURCES = (("phase3", "attention/gradient"), ("phase3b", "attention/gradient_3b"))


def sources(cfg: dict):
    declared = cfg["outputs"].get("attention_sources")
    if not declared:
        return DEFAULT_SOURCES
    return tuple((str(d["tag"]), str(d["dir"])) for d in declared)


def bag_stats(alpha: np.ndarray) -> dict:
    n = int(alpha.shape[0])
    top = max(1, n // 10)
    return {
        "n_patches": n,
        "max_over_uniform": float(alpha.max() * n),          # 1.0 == perfectly uniform
        "max_over_min": float(alpha.max() / alpha.min()),
        "normalized_entropy": float(-(alpha * np.log(alpha)).sum() / np.log(n)),
        "top10pct_mass": float(np.sort(alpha)[::-1][:top].sum()),
    }


def collect(root: str, rel_dir: str) -> pd.DataFrame:
    d = os.path.join(root, *rel_dir.split("/"))
    rows = []
    for fn in sorted(f for f in os.listdir(d) if f.endswith(".npz")):
        with np.load(os.path.join(d, fn), allow_pickle=True) as z:
            row = {"PatientID": str(z["patient_id"]),
                   "histology": str(z["histology"]),
                   "true_label": int(z["true_label"]),
                   "outer_fold": int(z["outer_fold"]),
                   "predicted_probability_ADC": float(z["predicted_probability_adc"])}
            row.update(bag_stats(np.asarray(z["attention"], dtype=np.float64)))
            rows.append(row)
    return pd.DataFrame(rows)


def concentration_by_fold(df: pd.DataFrame, keys, selected: dict) -> dict:
    """Median concentration per outer fold, next to the epoch that fold selected."""
    out = {}
    for f, sub in df.groupby("outer_fold"):
        row = {k: float(sub[k].median()) for k in keys}
        row["n_bags"] = int(len(sub))
        if selected:
            row["selected_epoch_attention_mil"] = selected.get(str(int(f)),
                                                               selected.get(int(f)))
        out[int(f)] = row
    return out


def plot_epoch_curves(cfg: dict) -> None:
    sel_dir = os.path.join(ppath(cfg, cfg["outputs"]["results_dir"]), "epoch_selection")
    if not os.path.isdir(sel_dir):
        return
    for m in MODELS:
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        for fold in range(1, 6):
            path = os.path.join(sel_dir, "%s_fold_%d.csv" % (m, fold))
            if not os.path.exists(path):
                continue
            c = pd.read_csv(path)
            line, = ax.plot(c["epoch"], c["mean_val_roc_auc"], lw=1.1,
                            label="outer fold %d" % fold)
            best = int(c["mean_val_roc_auc"].idxmax())
            ax.plot(c["epoch"][best], c["mean_val_roc_auc"][best], "o",
                    color=line.get_color(), ms=5)
        ax.axhline(0.5, color="k", ls="--", lw=0.8)
        ax.set_xlabel("epoch")
        ax.set_ylabel("mean inner-CV validation ROC-AUC")
        ax.set_title("Epoch selection - %s (3-fold inner CV)" % m)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(sel_dir, "%s_mean_curves.png" % m), dpi=150)
        plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="attention concentration statistics")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    args = ap.parse_args()
    cfg = load_phase3_config(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        *args.config.split("/")))
    root = cfg["_project_root"]
    out_dir = ppath(cfg, cfg["outputs"]["results_dir"])
    os.makedirs(out_dir, exist_ok=True)

    run_path = ppath(cfg, str(cfg["outputs"].get("run_json", "reports/phase3b_run.json")))
    run = (json.load(open(run_path, "r", encoding="utf-8"))
           if os.path.exists(run_path) else {})
    selected = (run.get("selected_epochs") or {}).get("attention_mil", {})

    summary = {}
    keys = ("max_over_uniform", "max_over_min", "normalized_entropy", "top10pct_mass")
    for tag, rel in sources(cfg):
        if not os.path.isdir(os.path.join(root, *rel.split("/"))):
            continue
        df = collect(root, rel)
        df.to_csv(os.path.join(out_dir, "attention_concentration_%s.csv" % tag), index=False)
        # A phase with its own results directory may additionally ask for the
        # un-suffixed file name.  Off by default, so earlier phases are unchanged.
        if bool(cfg["outputs"].get("attention_concentration_plain_csv", False)):
            df.to_csv(os.path.join(out_dir, "attention_concentration.csv"), index=False)
        summary[tag] = {
            "n_bags": int(len(df)),
            "overall": {k: {"mean": float(df[k].mean()), "median": float(df[k].median()),
                            "min": float(df[k].min()), "max": float(df[k].max())}
                        for k in keys},
            "by_fold": concentration_by_fold(df, keys, selected),
        }
        print("%s: %d bags, median normalized entropy %.4f, median max/uniform %.3f"
              % (tag, len(df), df["normalized_entropy"].median(),
                 df["max_over_uniform"].median()))

    if run:
        summary["selected_epochs"] = run.get("selected_epochs")
    with open(os.path.join(out_dir, "attention_concentration.json"), "w",
              encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    plot_epoch_curves(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
