"""Memory-frugal dense linear algebra for item-item models on small machines."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from scipy.linalg import lapack


def gram(
    X: sp.csr_matrix, dtype: type[np.floating] = np.float64, rows_per_block: int = 20_000
) -> NDArray[np.floating]:
    """Dense ``X^T X`` accumulated over row blocks.

    The full sparse product of a large log can hold far more non-zeros than the dense result, so
    we add each block's sparse product into one preallocated dense buffer. Peak memory is the
    dense n_items^2 buffer plus one block product.
    """
    X = X.tocsr().astype(dtype)
    n = X.shape[1]
    G = np.zeros((n, n), dtype=dtype)
    for s in range(0, X.shape[0], rows_per_block):
        Xb = X[s : s + rows_per_block]
        Pb = (Xb.T.tocsr() @ Xb).tocoo()  # coordinates are unique after the product
        G[Pb.row, Pb.col] += Pb.data
    return G


def spd_inverse_inplace(G: NDArray[np.floating], block: int = 2048) -> NDArray[np.floating]:
    """Invert a symmetric positive-definite matrix in place via Cholesky (xPOTRF + xPOTRI).

    Peak memory is one n x n matrix, versus three for ``np.linalg.inv``. The returned array is G's
    buffer, fully symmetrised.
    """
    if G.dtype == np.float64:
        potrf, potri = lapack.dpotrf, lapack.dpotri
    elif G.dtype == np.float32:
        potrf, potri = lapack.spotrf, lapack.spotri
    else:
        raise TypeError(f"unsupported dtype {G.dtype}")
    # LAPACK is column-major; a C-contiguous symmetric matrix viewed as its transpose is
    # F-contiguous with the same content, so we can work on G.T without copying.
    Gf = G.T
    c, info = potrf(Gf, lower=0, overwrite_a=1, clean=0)
    if info != 0:
        raise np.linalg.LinAlgError(f"potrf failed (info={info}); matrix not positive definite")
    inv, info = potri(c, lower=0, overwrite_c=1)
    if info != 0:
        raise np.linalg.LinAlgError(f"potri failed (info={info})")
    P = np.asarray(inv).T  # back to the C-order view of the same buffer
    # Only the upper triangle (in Fortran terms) is valid: mirror it block by block.
    n = P.shape[0]
    for s in range(0, n, block):
        e = min(n, s + block)
        # In the C-order view the valid part is the lower triangle; copy it to the upper one.
        P[s:e, e:] = P[e:, s:e].T
        tri = P[s:e, s:e]
        iu = np.triu_indices(e - s, 1)
        tri[iu] = tri.T[iu]
    return P
