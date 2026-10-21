"""EASE: Embarrassingly Shallow Autoencoder (Steck, WWW 2019).

Closed form: B = I - P diagMat(1 / diag(P)), with P = (X^T X + lambda I)^-1, which enforces
diag(B) = 0 exactly. Fitting is one dense SPD inversion of the item Gram matrix.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from mitate.linalg import gram, spd_inverse_inplace


def ease_weights(
    X: sp.csr_matrix, lam: float, dtype: type[np.floating] = np.float64
) -> NDArray[np.floating]:
    G = gram(X, dtype)
    G[np.diag_indices_from(G)] += lam
    P = spd_inverse_inplace(G)  # G's buffer now holds P
    d = np.diag(P).copy()
    P /= -d[None, :]
    P[np.diag_indices_from(P)] = 0.0
    return P


def ease_scores(
    X: sp.csr_matrix, B: NDArray[np.floating], out_dtype: type[np.floating] = np.float32
) -> NDArray[np.floating]:
    return np.asarray(X.astype(B.dtype) @ B, dtype=out_dtype)
