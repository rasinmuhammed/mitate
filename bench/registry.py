"""Name -> model class for the benchmark: the engine's table plus benchmark-only baselines.

Benchmark-only models (non-commercial code, ledger D6) live in ``bench/`` and are never reachable
from the engine or the server.
"""

from bench.gru4rec import GRU4Rec
from mitate.models.base import Recommender
from mitate.models.registry import MODELS as ENGINE_MODELS
from mitate.models.registry import NEURAL as ENGINE_NEURAL

MODELS: dict[str, type[Recommender]] = {**ENGINE_MODELS, GRU4Rec.name: GRU4Rec}
NEURAL = ENGINE_NEURAL | {GRU4Rec.name}

__all__ = ["MODELS", "NEURAL"]
