"""Record the exact environment used for an extraction run."""
from __future__ import annotations

import json
import os
import platform
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config_io import (DEFAULT_CONFIG, extraction_signature,  # noqa: E402
                       load_config, load_radiomics_params, project_path)


def collect(cfg):
    import matplotlib, nibabel, numpy, pandas, radiomics, scipy, sklearn
    import SimpleITK as sitk
    import yaml
    return {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python_version": sys.version,
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "packages": {
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
            "pandas": pandas.__version__,
            "scikit-learn": sklearn.__version__,
            "SimpleITK": sitk.__version__,
            "pyradiomics": radiomics.__version__,
            "matplotlib": matplotlib.__version__,
            "nibabel": nibabel.__version__,
            "PyYAML": yaml.__version__,
        },
        "seeds": {"numpy_random_state_for_splits": cfg["validation"]["random_state"],
                  "global_seed": cfg["reproducibility"]["seed"]},
        "config_path": cfg["_config_path"],
        "extraction_signature": extraction_signature(cfg),
        "experiment_config": {k: v for k, v in cfg.items() if not k.startswith("_")},
        "radiomics_params": load_radiomics_params(cfg),
    }


if __name__ == "__main__":
    cfg = load_config(DEFAULT_CONFIG)
    info = collect(cfg)
    out = project_path(cfg, "reports", "environment.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2)
    print(json.dumps({k: info[k] for k in
                      ["python_version", "platform", "cpu_count", "packages",
                       "extraction_signature"]}, indent=2))
    print("written:", out)
