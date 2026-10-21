"""Copy the serving rankers from the OTTO benchmark into the package (research ledger D25).

    uv run python scripts/ship_rankers.py

Only OTTO (CC BY 4.0) may produce anything the product ships, so the source is fixed. The
serving rankers are ``full`` and ``no_<source>`` (mitate.engine.ranker_sources); research
ablations such as ``no_volume`` stay in the benchmark. ``MANIFEST.json`` records where each
file came from: the stores it was trained on, the code version and a checksum.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from mitate.engine import SHIPPED_RANKERS, ranker_sources
from mitate.ltr import TreeEnsemble

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results" / "otto"
ATTRIBUTION = "OTTO session dataset, otto/recsys-dataset on Kaggle, CC BY 4.0"


def main() -> None:
    src = RESULTS / "_ltr"
    serving: dict[Path, set[str]] = {}
    for path in sorted(src.glob("*.json")):
        sources = ranker_sources(path.stem, TreeEnsemble.load(path).feature_names)
        if sources is not None:
            serving[path] = sources
    if not any(p.stem == "full" for p in serving):
        sys.exit(f"no full ranker in {src}: run OTTO stage 2 first")
    stores = [r for r in json.loads((RESULTS / "stores.json").read_text()) if "skipped" not in r]
    code = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "kaggle" / "code_version.py")],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if SHIPPED_RANKERS.exists():
        shutil.rmtree(SHIPPED_RANKERS)
    SHIPPED_RANKERS.mkdir(parents=True)
    files = {}
    for path, sources in serving.items():
        shutil.copy2(path, SHIPPED_RANKERS / path.name)
        files[path.name] = {
            "sources": sorted(sources),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    manifest = {
        "trained_on": ATTRIBUTION,
        "stores": [r["store"] for r in stores],
        "code_version": code,
        "exported_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rankers": files,
    }
    (SHIPPED_RANKERS / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    print(f"shipped {sorted(files)} from {len(stores)} OTTO stores to {SHIPPED_RANKERS}")


if __name__ == "__main__":
    main()
