"""Name -> model class. Every benchmarked model is reachable by its ``name``."""

from __future__ import annotations

from mitate.content import ContentKNN
from mitate.models.base import Recommender
from mitate.models.baselines import EASE, GPTopFreq, ItemKNN, Popularity, RecencyPopularity, RP3beta
from mitate.models.mf import BPR, IALS
from mitate.models.sasrec import ESASRec, GSASRec, SASRecCE
from mitate.models.sequential import PCTM, MarkovChain, SeqRules
from mitate.models.sknn import VSTAN

MODELS: dict[str, type[Recommender]] = {
    c.name: c
    for c in (
        Popularity,
        RecencyPopularity,
        GPTopFreq,
        ItemKNN,
        RP3beta,
        EASE,
        IALS,
        BPR,
        MarkovChain,
        PCTM,
        SeqRules,
        VSTAN,
        SASRecCE,
        GSASRec,
        ESASRec,
        ContentKNN,
    )
}
NEURAL = {"sasrec_ce", "gsasrec", "esasrec"}
