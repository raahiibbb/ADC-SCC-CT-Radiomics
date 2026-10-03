"""Read-only guarantee for the two external input trees.

CLAUDE.md forbids modifying anything under
    D:\\User\\Documents\\L.U.N.G\\Dataset
    D:\\User\\Documents\\L.U.N.G\\Vuong5ROI

A baseline manifest (path -> size, mtime) is recorded once, and `verify` re-walks
the same paths and reports any file that was added, removed, resized or touched.

Scope: the Vuong5ROI full_cohort output tree that the pipeline reads, plus the
individual Dataset metadata files it reads.  The full DICOM tree is not walked
(hundreds of thousands of files); the pipeline never opens it except for the
optional DICOM HU cross-check, which is read-only by construction.

    python src/readonly_guard.py snapshot
    python src/readonly_guard.py verify
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config_io import DEFAULT_CONFIG, load_config, project_path   # noqa: E402

MANIFEST = "reports/readonly_baseline.json"


def _targets(cfg):
    trees = [cfg["paths"]["vuong_root"]]
    files = [cfg["paths"]["dataset_metadata"], cfg["paths"]["cohort_summary"]]
    doc = os.path.join(os.path.dirname(cfg["paths"]["dataset_metadata"]),
                       "DATASET_DOCUMENTATION.md")
    if os.path.isfile(doc):
        files.append(doc)
    return trees, files


def _scan(cfg):
    trees, files = _targets(cfg)
    out = {}
    for t in trees:
        for dp, _, fns in os.walk(t):
            for fn in fns:
                p = os.path.join(dp, fn)
                try:
                    st = os.stat(p)
                except OSError:
                    continue
                out[os.path.normcase(p)] = [st.st_size, round(st.st_mtime, 3)]
    for p in files:
        try:
            st = os.stat(p)
            out[os.path.normcase(p)] = [st.st_size, round(st.st_mtime, 3)]
        except OSError:
            pass
    return out


def snapshot(cfg):
    entries = _scan(cfg)
    path = project_path(cfg, *MANIFEST.split("/"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    newest = max((v[1] for v in entries.values()), default=0)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "roots": _targets(cfg)[0],
            "extra_files": _targets(cfg)[1],
            "n_files": len(entries),
            "newest_mtime_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(newest)),
            "entries": entries,
        }, fh)
    print("baseline: %d files -> %s" % (len(entries), path))
    return path


def verify(cfg):
    path = project_path(cfg, *MANIFEST.split("/"))
    with open(path, encoding="utf-8") as fh:
        base = json.load(fh)["entries"]
    now = _scan(cfg)
    added = sorted(set(now) - set(base))
    removed = sorted(set(base) - set(now))
    changed = sorted(k for k in set(base) & set(now) if base[k] != now[k])
    ok = not (added or removed or changed)
    return ok, {"n_baseline": len(base), "n_now": len(now), "added": added[:20],
                "removed": removed[:20], "changed": changed[:20],
                "n_added": len(added), "n_removed": len(removed),
                "n_changed": len(changed)}


if __name__ == "__main__":
    cfg = load_config(DEFAULT_CONFIG)
    cmd = sys.argv[1] if len(sys.argv) > 1 else "verify"
    if cmd == "snapshot":
        snapshot(cfg)
    else:
        ok, info = verify(cfg)
        print(json.dumps(info, indent=2))
        print("READ-ONLY GUARANTEE:", "INTACT" if ok else "VIOLATED")
        sys.exit(0 if ok else 1)
