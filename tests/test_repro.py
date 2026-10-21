"""Published-number reproductions (Gate 2). Skipped when the raw data is not downloaded."""

from pathlib import Path

import pytest

pytestmark = pytest.mark.slow
NBR = Path("data/raw/nbr_reality_check")


@pytest.mark.skipif(not (NBR / "instacart_history.csv").exists(), reason="data not downloaded")
def test_gp_topfreq_instacart_matches_reality_check_code() -> None:
    from bench.repro.gp_topfreq_instacart import EXPECTED, main

    got = main(str(NBR), tiebreak="first")
    for k, v in EXPECTED.items():
        assert abs(got[k] - v) <= 5e-4, (k, got[k], v)
