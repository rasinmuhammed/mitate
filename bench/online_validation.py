"""Known-truth validation of the online measurement module (docs/results.md, "Online").

    python -m bench.online_validation [--out experiments/results/online/validation.json]

1. A/B estimator (raw and CUPED): bias against each draw's exact population lift, CI coverage,
   variance reduction, and the false-positive rate at zero lift, in two traffic regimes.
2. Power calculator: the normal-approximation sample size is checked against simulated power on
   heavy-tailed revenue, at the sample size the calculator asks for.
3. Team-draft interleaving: power to prefer the truly better ranker, wrong-direction errors, and
   size/coverage under the null, at small-store visitor counts.

Everything is seeded; the output records the parameters next to every number.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from mitate.online.abtest import required_visitors, simulate_power
from mitate.online.simulate import REGIMES, ab_validation, interleaving_validation, population


def power_check(regime: str, rel_mde: float, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    rate, visits = REGIMES[regime]
    pop = population(200_000, rng, rate, visits)
    y = rng.binomial(visits, pop.base_conv) * pop.order_value
    n = required_visitors(float(y.mean()), float(y.std()), rel_mde)
    return {
        "regime": regime,
        "rel_mde": rel_mde,
        "visitors_required_formula": n,
        "simulated_power_at_that_n": simulate_power(y, rel_mde, n, n_sims=1000, seed=seed),
        "target_power": 0.8,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/results/online/validation.json")
    ap.add_argument("--sims", type=int, default=1000)
    ap.add_argument("--il-sims", type=int, default=200)
    a = ap.parse_args()
    t0 = time.monotonic()
    out: dict[str, Any] = {"ab": [], "power": [], "interleaving": []}
    res: dict[str, Any]  # A/B and interleaving results differ in shape
    for regime in REGIMES:
        for n in (5_000, 20_000):
            res = ab_validation(n=n, lift=0.10, sims=a.sims, seed=0, regime=regime)
            row = {"regime": regime, "n_visitors": n, "lift": 0.10, "sims": a.sims}
            out["ab"].append(row | res)
            print(regime, n, json.dumps(res["cuped"]), flush=True)
        for mde in (0.05, 0.10):
            out["power"].append(power_check(regime, mde, seed=1))
            print(out["power"][-1], flush=True)
    for n in (300, 1000, 3000):
        res = interleaving_validation(n_visitors=n, quality_gap=0.3, sims=a.il_sims, seed=2)
        out["interleaving"].append(
            {"n_visitors": n, "quality_gap": 0.3, "bot_share": 0.01, "sims": a.il_sims, **res}
        )
        print(n, res, flush=True)
    out["seconds"] = time.monotonic() - t0
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
