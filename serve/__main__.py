"""``python -m serve run`` | ``create-merchant NAME`` | ``retrain MERCHANT_ID``."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from serve.app import create_app, engine_config
from serve.db import DB
from serve.service import Service


def _service() -> Service:
    d = Path(os.environ.get("MITATE_DATA_DIR", "data/serve"))
    d.mkdir(parents=True, exist_ok=True)
    cfg = engine_config(os.environ.get("MITATE_ENGINE_CONFIG"))
    return Service(DB(d / "mitate.sqlite"), d, cfg, background=False)


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m serve")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--host", default="127.0.0.1")
    r.add_argument("--port", type=int, default=8000)
    c = sub.add_parser("create-merchant")
    c.add_argument("name")
    t = sub.add_parser("retrain")
    t.add_argument("merchant")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    if a.cmd == "run":
        import uvicorn

        # access_log=False: uvicorn's access log prints full URLs; ours logs route templates
        uvicorn.run(create_app(), host=a.host, port=a.port, access_log=False)
    elif a.cmd == "create-merchant":
        keys = _service().create_merchant(a.name)
        print(json.dumps(keys, indent=1))
        print("Store the private key now; it is not shown again (only its hash is kept).")
    else:
        s = _service()
        print(s.retrain(a.merchant))


if __name__ == "__main__":
    main()
