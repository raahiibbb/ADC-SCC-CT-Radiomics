"""Phase 8B - progress / status reporter for the long nested run.

Safe to run at any time, including while src/phase8b_run.py is executing: it
only reads.  Prints which units are complete, which remain, the elapsed time
and a projected finish.

    ./.venv/Scripts/python.exe src/phase8b_status.py --config config/phase8b_multicenter.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from phase8b_data import load_cfg, ppath, protocol_sha, sha256   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/phase8b_multicenter.yaml")
    args = ap.parse_args()
    cfg = load_cfg(args.config)

    units = ["pooled_fold_%d" % k
             for k in range(1, int(cfg["validation"]["outer"]["n_splits"]) + 1)]
    units += ["transfer_%s" % d["id"] for d in cfg["validation"]["transfer"]["directions"]]
    units += ["domain_diagnostic"]

    psha = protocol_sha(cfg)
    run_path = ppath(cfg, "run_json")
    print("Phase 8B status")
    print("  protocol sha256 : %s" % psha[:32])
    if os.path.isfile(run_path):
        rec = json.load(open(run_path, encoding="utf-8"))
        fb = rec.get("frozen_before_modelling", {})
        print("  frozen           : %s  (protocol %s)"
              % (rec.get("frozen_at"), fb.get("protocol_sha256", "")[:16]))
        print("  protocol intact  : %s" % (fb.get("protocol_sha256") == psha))
        print("  config intact    : %s"
              % (fb.get("config_sha256") == sha256(cfg["_config_path"])))
        print("  modelling done   : %s" % ("modelling" in rec))
        print("  evaluation done  : %s" % ("evaluation" in rec))
    else:
        print("  NOT FROZEN - run  src/phase8b_data.py --freeze  first")

    ck_dir = ppath(cfg, "checkpoints_dir")
    done, pending = [], []
    for u in units:
        p = os.path.join(ck_dir, "%s.json" % u)
        ok = False
        if os.path.isfile(p):
            try:
                d = json.load(open(p, encoding="utf-8"))
                ok = bool(d.get("complete") and d.get("protocol_sha256") == psha)
            except (ValueError, OSError):
                ok = False
        (done if ok else pending).append(u)

    print("\n  units complete   : %d / %d" % (len(done), len(units)))
    for u in units:
        print("    [%s] %s" % ("x" if u in done else " ", u))

    pj = ppath(cfg, "progress_json")
    if os.path.isfile(pj):
        pr = json.load(open(pj, encoding="utf-8"))
        el = float(pr.get("elapsed_seconds", 0.0))
        nd = int(pr.get("units_done", 0))
        print("\n  current unit     : %s" % pr.get("current_unit"))
        print("  elapsed          : %.0f s (%.1f min), updated %s"
              % (el, el / 60.0, pr.get("updated")))
        age = time.time() - os.path.getmtime(pj)
        print("  progress file age: %.0f s %s"
              % (age, "(run appears finished or stalled)" if age > 900 else "(running)"))
        if nd and nd < len(units):
            rate = el / nd
            print("  projected finish : ~%.1f more min"
                  % (rate * (len(units) - nd) / 60.0))

    if pending:
        print("\n  RESUME with the same command - complete units are skipped:")
        print("    ./.venv/Scripts/python.exe src/phase8b_run.py --config %s --resume"
              % os.path.relpath(cfg["_config_path"], cfg["_project_root"]).replace(os.sep, "/"))
    else:
        print("\n  all modelling units complete -> run src/phase8b_evaluate.py")


if __name__ == "__main__":
    main()
