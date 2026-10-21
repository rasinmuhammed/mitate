"""OTTO (CC BY 4.0) ingestion: zipped JSONL -> one flat Parquet file per split.

The 11 GB train JSONL is never written to disk: ``unzip -p`` streams it, lines are batched into
bounded chunks, and polars' native NDJSON reader parses each chunk. Output columns:

    session int32, aid int32, ts int64 (Unix seconds), type uint8 (0 click, 1 cart, 2 order)

OTTO timestamps are documented as Unix time; the 13-digit values are milliseconds, so they are
divided by 1000 (checked at load time against the documented 4+1 week span).

Usage: python -m bench.data.otto data/raw/otto data/otto
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path
from typing import cast

import polars as pl
import pyarrow.parquet as pq

TYPES = {"clicks": 0, "carts": 1, "orders": 2}
LINES_PER_CHUNK = 200_000


def _convert_chunk(buf: bytes) -> pl.DataFrame:
    df = pl.read_ndjson(
        io.BytesIO(buf),
        schema={
            "session": pl.Int64,
            "events": pl.List(pl.Struct({"aid": pl.Int64, "ts": pl.Int64, "type": pl.Utf8})),
        },
    )
    return (
        df.explode("events")
        .unnest("events")
        .select(
            pl.col("session").cast(pl.Int32),
            pl.col("aid").cast(pl.Int32),
            (pl.col("ts") // 1000).alias("ts"),
            pl.col("type").replace_strict(TYPES, return_dtype=pl.UInt8),
        )
    )


def convert(zip_path: Path, member: str, dest: Path) -> dict[str, object]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".partial")
    proc = subprocess.Popen(["unzip", "-p", str(zip_path), member], stdout=subprocess.PIPE)
    assert proc.stdout is not None
    writer: pq.ParquetWriter | None = None
    n_sessions = n_events = 0
    ts_min, ts_max = 1 << 62, 0
    lines: list[bytes] = []

    def flush() -> None:
        nonlocal writer, n_sessions, n_events, ts_min, ts_max
        if not lines:
            return
        df = _convert_chunk(b"".join(lines))
        n_sessions += len(lines)
        n_events += df.height
        ts_min = min(ts_min, int(cast(int, df["ts"].min())))  # integer column
        ts_max = max(ts_max, int(cast(int, df["ts"].max())))
        t = df.to_arrow()
        if writer is None:
            writer = pq.ParquetWriter(tmp, t.schema, compression="zstd")
        writer.write_table(t)
        lines.clear()
        print(f"  {dest.name}: {n_sessions:,} sessions, {n_events:,} events", flush=True)

    for line in proc.stdout:
        lines.append(line)
        if len(lines) >= LINES_PER_CHUNK:
            flush()
    flush()
    if proc.wait() != 0:
        raise RuntimeError(f"unzip failed for {zip_path}")
    assert writer is not None
    writer.close()
    tmp.rename(dest)
    span_days = (ts_max - ts_min) / 86_400
    if not 4 <= span_days <= 40:
        raise ValueError(f"unexpected time span {span_days:.1f} days; timestamp unit wrong?")
    return {
        "source": zip_path.name,
        "member": member,
        "sessions": n_sessions,
        "events": n_events,
        "ts_min": ts_min,
        "ts_max": ts_max,
    }


def main(raw: str, out: str) -> None:
    raw_dir, out_dir = Path(raw), Path(out)
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for split in ("train", "test"):
        z = raw_dir / f"otto-recsys-{split}.jsonl.zip"
        dest = out_dir / f"{split}.parquet"
        if dest.exists() or not z.exists():
            continue
        manifest[split] = convert(z, f"otto-recsys-{split}.jsonl", dest)
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
