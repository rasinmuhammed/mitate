"""SASRec family in plain PyTorch.

* ``loss="ce"``: full-softmax cross-entropy, i.e. SASRec+ (Klenitskiy & Vasilev, RecSys 2023).
  Affordable here because small-store catalogues are small.
* ``loss="gbce"``: gSASRec (Petrov & Macdonald, RecSys 2023) with k uniform negatives and the
  calibrated beta from t.
* ``loss="ssm"``: sampled softmax with k uniform negatives.
* ``blocks="ligr"``: LiGR gated pre-norm layers with SwiGLU FFN (as in RecTools' LiGRLayers,
  Apache-2.0); with ``loss="ssm"`` this is the eSASRec recipe (Tikhonovich et al., RecSys 2025).

Training uses the shifted-sequence objective on every position. Long sequences are cut into
non-overlapping windows from the end. The number of epochs is a hyperparameter: during tuning a
``monitor`` callback scores validation after each epoch and the best epoch is recorded so the final
retrain on train+validation runs exactly that many epochs.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar, Self, cast

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from mitate.data import I32, I64, Catalog, Log, Queries
from mitate.models.base import F32, Recommender
from mitate.models.sequential import training_sequences


class SwiGLU(nn.Module):
    def __init__(self, d: int, mult: int, dropout: float) -> None:
        super().__init__()
        self.w12 = nn.Linear(d, 2 * mult * d, bias=False)
        self.w3 = nn.Linear(mult * d, d, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.w12(x).chunk(2, dim=-1)
        return cast(torch.Tensor, self.w3(self.drop(F.silu(a) * b)))


class Block(nn.Module):
    def __init__(self, d: int, heads: int, dropout: float, ligr: bool) -> None:
        super().__init__()
        self.ligr = ligr
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.drop1, self.drop2 = nn.Dropout(dropout), nn.Dropout(dropout)
        if ligr:
            self.ff: nn.Module = SwiGLU(d, 4, dropout)
            self.g1, self.g2 = nn.Linear(d, d), nn.Linear(d, d)
        else:
            self.ff = nn.Sequential(
                nn.Linear(d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, d)
            )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + (torch.sigmoid(self.g1(x)) if self.ligr else 1.0) * self.drop1(a)
        f = self.ff(self.ln2(x))
        x = x + (torch.sigmoid(self.g2(x)) if self.ligr else 1.0) * self.drop2(f)
        return x


class Net(nn.Module):
    def __init__(
        self,
        n_items: int,
        d: int,
        n_blocks: int,
        heads: int,
        max_len: int,
        dropout: float,
        ligr: bool,
        tied: bool = True,
    ) -> None:
        super().__init__()
        self.item = nn.Embedding(n_items + 1, d, padding_idx=0)  # id 0 = padding
        # Untied output embeddings (gSASRec's reference default): with BCE, hundreds of sampled
        # negatives per position would otherwise drag down the *input* embeddings of every item.
        self.out = None if tied else nn.Embedding(n_items, d)
        self.pos = nn.Embedding(max_len, d)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(d, heads, dropout, ligr) for _ in range(n_blocks)])
        self.ln = nn.LayerNorm(d)
        self.max_len = max_len
        nn.init.normal_(self.item.weight, std=d**-0.5)
        if self.out is not None:
            nn.init.normal_(self.out.weight, std=d**-0.5)
        with torch.no_grad():
            self.item.weight[0].zero_()

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        L = seq.shape[1]
        pos = torch.arange(L, device=seq.device)
        x = self.item(seq) * math.sqrt(self.item.embedding_dim) + self.pos(pos)[None]
        x = self.drop(x)
        pad = seq == 0
        # Causal + key padding, but every position may attend to itself so left-padded rows never
        # see an all-masked softmax (which would produce NaN that leaks through 0 * NaN).
        eye = torch.eye(L, dtype=torch.bool, device=seq.device)
        causal = torch.ones(L, L, dtype=torch.bool, device=seq.device).triu(1)
        mask = (causal[None] | pad[:, None, :]) & ~eye[None]
        mask = mask.repeat_interleave(cast(Block, self.blocks[0]).attn.num_heads, dim=0)
        for b in self.blocks:
            x = b(x, mask)
            x = x.masked_fill(pad[..., None], 0.0)
        return cast(torch.Tensor, self.ln(x))

    def output_weights(self) -> torch.Tensor:
        return self.item.weight[1:] if self.out is None else self.out.weight

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        return h @ self.output_weights().T  # (…, n_items), item i -> column i


def _windows(seq: I64, items: I32, max_len: int) -> list[I32]:
    """Per training sequence, non-overlapping windows of max_len+1 cut from the end."""
    out: list[I32] = []
    bounds = np.flatnonzero(np.diff(seq)) + 1
    for chunk in np.split(items, bounds):
        if chunk.size < 2:
            continue
        end = chunk.size
        while end >= 2:
            start = max(0, end - (max_len + 1))
            out.append(chunk[start:end])
            if start == 0:
                break
            end = start + 1  # overlap one item so every transition is used once
    return out


@dataclass
class SASRec(Recommender):
    name: ClassVar[str] = "sasrec"
    loss: str = "ce"  # ce | gbce | ssm
    blocks: str = "std"  # std | ligr
    d: int = 64
    n_blocks: int = 2
    heads: int = 2
    max_len: int = 50
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 128
    epochs: int = 30
    negatives: int = 256
    gbce_t: float = 0.75
    tied: bool = True  # share input and output item embeddings
    patience: int = 5
    device: str = "cpu"
    threads: int = 4
    time_budget_s: float = 3600.0
    monitor: Callable[[Recommender], float] | None = field(default=None, repr=False)
    best_epoch: int = field(default=0, repr=False)
    net: Net | None = field(default=None, repr=False)

    @classmethod
    def space(cls, trial: Any) -> dict[str, Any]:
        return {
            "d": trial.suggest_categorical("d", [32, 64, 128]),
            "n_blocks": trial.suggest_int("n_blocks", 1, 3),
            "heads": trial.suggest_categorical("heads", [1, 2]),
            "max_len": trial.suggest_categorical("max_len", [20, 50, 100]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True),
        }

    def _batch_loss(self, net: Net, x: torch.Tensor, y: torch.Tensor, n: int) -> torch.Tensor:
        h: torch.Tensor = net(x)
        mask = y > 0
        h, tgt = h[mask], y[mask] - 1
        if self.loss == "ce":
            return F.cross_entropy(net.logits(h), tgt)
        k = min(self.negatives, max(n - 1, 1))
        neg = torch.randint(0, n, (h.shape[0], k), device=h.device)
        W = net.output_weights()
        pos_s = (h * W[tgt]).sum(-1)
        neg_s = torch.einsum("bd,bkd->bk", h, W[neg])
        neg_s = neg_s.masked_fill(neg == tgt[:, None], -1e9)  # accidental positives
        if self.loss == "ssm":
            return F.cross_entropy(torch.cat([pos_s[:, None], neg_s], 1), torch.zeros_like(tgt))
        # gBCE (Petrov & Macdonald, RecSys 2023, Eq. 10): -beta*log s(pos) - sum log(1 - s(neg)),
        # written with softplus so it is stable at any logit. (The paper's equivalent form maps
        # the positive logit through sigmoid**beta; clamping that sigmoid kills the gradient of
        # strongly negative positives.) Averaged over the 1 + k entries like the reference BCE.
        alpha = k / max(n - 1, 1)
        beta = alpha * (self.gbce_t * (1 - 1 / alpha) + 1 / alpha)
        valid = neg_s > -1e8
        neg_l = torch.where(valid, F.softplus(neg_s), torch.zeros_like(neg_s)).sum()
        return (beta * F.softplus(-pos_s).sum() + neg_l) / (pos_s.numel() + valid.sum())

    def fit(self, log: Log, catalog: Catalog) -> Self:
        torch.manual_seed(self.seed)
        torch.set_num_threads(self.threads)
        rng = np.random.default_rng(self.seed)
        n = log.n_items
        seq, items = training_sequences(log)
        wins = _windows(seq, items + 1, self.max_len)  # shift ids: 0 is padding
        dev = torch.device(self.device)
        net = Net(
            n,
            self.d,
            self.n_blocks,
            self.heads,
            self.max_len,
            self.dropout,
            self.blocks == "ligr",
            self.tied,
        ).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.net = net
        if not wins:
            return self
        X = np.zeros((len(wins), self.max_len + 1), np.int64)
        for r, w in enumerate(wins):
            X[r, -w.size :] = w
        X_t = torch.from_numpy(X)
        best, best_state, bad = -math.inf, None, 0
        t0 = time.monotonic()
        max_epochs = self.epochs if self.monitor is None else max(self.epochs, 200)
        for epoch in range(1, max_epochs + 1):
            net.train()
            perm = torch.from_numpy(rng.permutation(len(wins)))
            for s in range(0, len(wins), self.batch_size):
                b = X_t[perm[s : s + self.batch_size]].to(dev)
                loss = self._batch_loss(net, b[:, :-1], b[:, 1:], n)
                opt.zero_grad(set_to_none=True)
                loss.backward()  # type: ignore[no-untyped-call]  # torch leaves it unannotated
                opt.step()
            if self.monitor is not None:
                net.eval()
                v = self.monitor(self)
                if v > best:
                    best, bad, self.best_epoch = v, 0, epoch
                    best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
                else:
                    bad += 1
                if bad >= self.patience:
                    break
            if time.monotonic() - t0 > self.time_budget_s:
                break
        if best_state is not None:
            net.load_state_dict(best_state)
        else:
            self.best_epoch = epoch
        net.eval()
        return self

    @torch.no_grad()
    def score(self, q: Queries) -> F32:
        assert self.net is not None
        n = self.net.item.num_embeddings - 1
        X = np.zeros((len(q), self.max_len), np.int64)
        for r in range(len(q)):
            h = q.history(r)
            if h.size == 0:
                continue
            keep = np.ones(h.size, bool)
            keep[1:] = h[1:] != h[:-1]
            h = h[keep][-self.max_len :] + 1
            X[r, -h.size :] = h
        dev = next(self.net.parameters()).device
        out = np.empty((len(q), n), np.float32)
        empty = (X == 0).all(axis=1)
        for s in range(0, len(q), 512):
            xb = torch.from_numpy(X[s : s + 512]).to(dev)
            h = self.net(xb)[:, -1]
            out[s : s + 512] = self.net.logits(h).float().cpu().numpy()
        out[empty] = 0.0  # no history: no opinion; the blend falls back to other sources
        return out


@dataclass
class SASRecCE(SASRec):
    """SASRec+ : full-softmax cross-entropy (Klenitskiy & Vasilev, RecSys 2023)."""

    name: ClassVar[str] = "sasrec_ce"
    loss: str = "ce"


@dataclass
class GSASRec(SASRec):
    """gSASRec: gBCE with uniform negatives (Petrov & Macdonald, RecSys 2023).

    The paper's defaults (k = 256 negatives, t = 0.75) assume catalogs of thousands of items. In a
    store with a few hundred items, k = 256 samples a third of the catalog per positive and the
    model only learns popularity (measured: research_ledger.md, D16), so k and t are tuned, as in
    the paper's own sensitivity study, inside the same trial budget as every other model.
    """

    name: ClassVar[str] = "gsasrec"
    loss: str = "gbce"

    @classmethod
    def space(cls, trial: Any) -> dict[str, Any]:
        return super().space(trial) | {
            "negatives": trial.suggest_categorical("negatives", [1, 4, 16, 64, 256]),
            "gbce_t": trial.suggest_float("gbce_t", 0.0, 1.0),
        }


@dataclass
class ESASRec(SASRec):
    """eSASRec recipe: LiGR layers + sampled softmax (Tikhonovich et al., RecSys 2025)."""

    name: ClassVar[str] = "esasrec"
    loss: str = "ssm"
    blocks: str = "ligr"
