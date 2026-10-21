"""Content hash of the code shipped to Kaggle, so a run can prove which code it mounted.

    python3 scripts/kaggle/code_version.py

Hashes every file under the paths that ``package.sh`` zips, in sorted order, by relative path
and contents. Standard library only: it runs before any environment exists.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

PATHS = (
    "mitate",
    "bench",
    "serve",
    "experiments/configs",
    "pyproject.toml",
    "uv.lock",
    ".python-version",
    "README.md",
)


def _shipped(rel: Path) -> bool:
    """Rankers the package ships are made from benchmark output, so the benchmark never reads
    them and they are no part of its code (research ledger D25)."""
    return rel.parts[:2] == ("mitate", "rankers")


def code_version(root: Path) -> str:
    files: list[Path] = []
    for rel in PATHS:
        p = root / rel
        if p.is_dir():
            files.extend(f for f in p.rglob("*") if f.is_file())
        else:
            files.append(p)
    h = hashlib.sha256()
    for f in sorted(files):
        if "__pycache__" in f.parts or f.suffix == ".pyc" or _shipped(f.relative_to(root)):
            continue
        h.update(f.relative_to(root).as_posix().encode() + b"\0")
        h.update(hashlib.sha256(f.read_bytes()).digest())
    return h.hexdigest()[:16]


if __name__ == "__main__":
    print(code_version(Path(__file__).resolve().parents[2]))
