"""GRU4Rec, official PyTorch implementation, as a benchmark-only baseline (ledger D6).

The official code is "free for research and education" only, so it is never vendored or shipped:
``official()`` downloads one pinned commit into the user cache and checks its SHA-256 before the
first import. Everything that defines the method is the official code: the model, both losses
(cross-entropy over softmax, BPR-max over ELU), popularity-based negative sampling, the
IndexedAdagradM optimiser, compatibility-mode initialisation and session-parallel mini-batches.

One shim keeps the pinned code byte-identical under NumPy 2: its compatibility-mode initialiser
is cast back to float32, as NumPy 1.x computed it. This adapter adds only what the protocol
requires of every neural model (protocol.md section 5):
an epoch loop with the shared validation monitor, patience and per-trial wall-clock cap (its body
is the body of the official ``GRU4Rec.fit``), and batched scoring of query histories. Sequences
are the ones SASRec trains on (``training_sequences``), and the search space is the union of the
official ``paramspaces/`` files, with the loss as a categorical choice.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, ClassVar, Self

import httpx
import numpy as np
import pandas as pd
import torch

from mitate.data import F32, Catalog, Log, Queries
from mitate.models.base import Recommender
from mitate.models.sequential import training_sequences

COMMIT = "d1fc31105577665d2f105de1c8d9a38d223a9a22"  # hidasib/GRU4Rec_PyTorch_Official, 2023-08-24
SHA256 = "d5b52620f76f36b13e229a53d11176d822d472270f3183fda25a034051e61278"
URL = f"https://raw.githubusercontent.com/hidasib/GRU4Rec_PyTorch_Official/{COMMIT}/gru4rec_pytorch.py"
CACHE = Path(os.environ.get("MITATE_CACHE", Path.home() / ".cache" / "mitate")) / "third_party"

_module: ModuleType | None = None


def official() -> ModuleType:
    """The pinned official module, downloaded once and verified on every load."""
    global _module
    if _module is not None:
        return _module
    path = CACHE / f"gru4rec-{COMMIT[:12]}" / "gru4rec_pytorch.py"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        r = httpx.get(URL, timeout=60, follow_redirects=True)
        r.raise_for_status()
        path.write_bytes(r.content)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SHA256:
        raise RuntimeError(f"{path}: SHA-256 {digest} does not match the pinned {SHA256}")
    spec = importlib.util.spec_from_file_location("gru4rec_official", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _module = mod
    return mod


@dataclass
class GRU4Rec(Recommender):
    name: ClassVar[str] = "gru4rec"
    loss: str = "cross-entropy"  # cross-entropy | bpr-max
    layers: int = 128
    batch_size: int = 64
    learning_rate: float = 0.05
    dropout_p_embed: float = 0.0
    dropout_p_hidden: float = 0.0
    momentum: float = 0.0
    sample_alpha: float = 0.5
    n_sample: int = 2048
    bpreg: float = 1.0
    elu_param: float = 0.5
    constrained_embedding: bool = True
    max_len: int = 50  # history steps fed at scoring time, as for SASRec
    epochs: int = 10
    patience: int = 5
    device: str = "cpu"  # the official sparse embeddings and optimiser are CPU/CUDA only
    threads: int = 4
    time_budget_s: float = 3600.0
    monitor: Callable[[Recommender], float] | None = field(default=None, repr=False)
    best_epoch: int = field(default=0, repr=False)
    model: Any = field(default=None, repr=False)
    item_index: Any = field(default=None, repr=False)  # official item index per dense item, -1

    @classmethod
    def space(cls, trial: Any) -> dict[str, Any]:
        p = {
            "loss": trial.suggest_categorical("loss", ["cross-entropy", "bpr-max"]),
            "layers": trial.suggest_int("layers", 64, 512, step=32),
            "batch_size": trial.suggest_int("batch_size", 32, 256, step=16),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.25, step=0.005),
            "dropout_p_embed": trial.suggest_float("dropout_p_embed", 0.0, 0.5, step=0.05),
            "dropout_p_hidden": trial.suggest_float("dropout_p_hidden", 0.0, 0.7, step=0.05),
            "momentum": trial.suggest_float("momentum", 0.0, 0.9, step=0.05),
            "sample_alpha": trial.suggest_float("sample_alpha", 0.0, 1.0, step=0.1),
        }
        if p["loss"] == "bpr-max":
            p["bpreg"] = trial.suggest_float("bpreg", 0.0, 2.0, step=0.05)
            p["elu_param"] = trial.suggest_categorical("elu_param", [0.5, 1.0, 0.0])
        return p

    def fit(self, log: Log, catalog: Catalog) -> Self:
        g = official()
        torch.manual_seed(self.seed)
        torch.set_num_threads(self.threads)
        dev = torch.device("cpu" if self.device == "mps" else self.device)
        key, items = training_sequences(log)
        data = pd.DataFrame(
            {"SessionId": key, "ItemId": items.astype(np.int64), "Time": np.arange(key.size)}
        )
        # session-parallel mini-batches hold one session per row, so the official iterator needs
        # at least as many training sessions as rows; small stores can have fewer than the batch
        net = g.GRU4Rec(
            layers=[self.layers],
            loss=self.loss,
            batch_size=min(self.batch_size, int(np.unique(key).size)),
            dropout_p_embed=self.dropout_p_embed,
            dropout_p_hidden=self.dropout_p_hidden,
            learning_rate=self.learning_rate,
            momentum=self.momentum,
            sample_alpha=self.sample_alpha,
            n_sample=self.n_sample,
            constrained_embedding=self.constrained_embedding,
            bpreg=self.bpreg,
            elu_param=self.elu_param,
            device=dev,
        )
        with contextlib.redirect_stdout(io.StringIO()):  # the official code prints progress
            it = g.SessionDataIterator(
                data,
                net.batch_size,
                n_sample=net.n_sample,
                sample_alpha=net.sample_alpha,
                sample_cache_max_size=2_000_000,
                session_order="time",
                device=dev,
            )
            model = g.GRU4RecModel(
                it.n_items,
                net.layers,
                net.dropout_p_embed,
                net.dropout_p_hidden,
                net.embedding,
                net.constrained_embedding,
            ).to(dev)
            # NumPy 2 promotes the official float32 init (times a float64 scale) to float64,
            # which the float32 parameters reject; NumPy 1.x kept float32. Same values, cast back.
            init = model._init_numpy_weights
            model._init_numpy_weights = lambda shape: init(shape).astype(np.float32)
            model._reset_weights_to_compatibility_mode()
        net.model = model
        index = np.full(log.n_items, -1, np.int64)
        index[it.itemidmap.index.to_numpy(np.int64)] = it.itemidmap.to_numpy(np.int64)
        self.model, self.item_index = model, index
        opt = g.IndexedAdagradM(model.parameters(), net.learning_rate, net.momentum)
        t0 = time.monotonic()
        best, bad, best_state = -np.inf, 0, None
        max_epochs = self.epochs if self.monitor is None else max(self.epochs, 100)
        for epoch in range(1, max_epochs + 1):
            model.train()
            H = [torch.zeros((net.batch_size, net.layers[0]), dtype=torch.float32, device=dev)]

            def reset(n_valid: int, finished: Any, valid: Any, H: list[Any] = H) -> bool:
                return bool(net._adjust_hidden(n_valid, finished, valid, H))

            n_valid = net.batch_size
            for in_idx, out_idx in it(enable_neg_samples=net.n_sample > 0, reset_hook=reset):
                for h in H:
                    h.detach_()
                model.zero_grad()
                R = model.forward(in_idx, H, out_idx, training=True)
                L = net.loss_function(R, out_idx, n_valid) / net.batch_size
                L.backward()
                opt.step()
                if not torch.isfinite(L):
                    raise FloatingPointError(f"GRU4Rec: non-finite loss in epoch {epoch}")
            model.eval()
            if self.monitor is not None:
                v = self.monitor(self)
                if v > best:
                    best, bad, self.best_epoch = v, 0, epoch
                    best_state = {k: t.clone() for k, t in model.state_dict().items()}
                else:
                    bad += 1
                if bad >= self.patience:
                    break
            else:
                self.best_epoch = epoch
            if time.monotonic() - t0 > self.time_budget_s:
                break
        if best_state is not None:
            model.load_state_dict(best_state)
        return self

    @torch.no_grad()
    def score(self, q: Queries) -> F32:
        assert self.model is not None and self.item_index is not None
        model, n = self.model, self.item_index.size
        dev = next(model.parameters()).device
        seqs = []
        for r in range(len(q)):
            h = q.history(r)
            keep = np.ones(h.size, bool)
            keep[1:] = h[1:] != h[:-1]
            idx = self.item_index[h[keep][-self.max_len :]]
            seqs.append(idx[idx >= 0])
        out = np.zeros((len(q), n), np.float32)
        known = self.item_index >= 0
        for s in range(0, len(q), 512):
            batch = seqs[s : s + 512]
            lens = np.array([x.size for x in batch])
            H = [torch.zeros((len(batch), model.layers[0]), device=dev)]
            last = torch.zeros((len(batch), model.layers[0]), device=dev)
            for t in range(int(lens.max(initial=0))):
                rows = np.flatnonzero(lens > t)
                X = torch.as_tensor(np.array([batch[r][t] for r in rows]), device=dev)
                Hs = [state[rows] for state in H]
                E, _, _ = model.embed(X, Hs, None)
                if not (model.constrained_embedding or model.embedding):
                    Hs[0] = E
                Xh = model.hidden_step(E, Hs, training=False)
                for state, new in zip(H, Hs, strict=True):
                    state[rows] = new
                last[rows] = Xh
            O = model.score_items(last, model.Wy.weight, model.By.weight).cpu().numpy()
            block = np.zeros((len(batch), n), np.float32)
            block[:, known] = O[:, self.item_index[known]]
            block[lens == 0] = 0.0  # no history: no opinion
            out[s : s + len(batch)] = block
        return out
