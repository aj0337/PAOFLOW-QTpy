import numpy as np
from typing import Tuple

# TODO This file is currently not used in the codebase. It should be removed or integrated into the main code.


def compute_dos(
    greens_function: np.ndarray,
    wk: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the total and k-resolved Density of States (DOS) from the Green's function.

    Parameters
    ----------
    `greens_function` : np.ndarray of shape (ne, nk, n, n)
        Green's function `G(E, k)` at each energy (`ne`) and k-point (`nk`).
        Each element `greens_function[ie, ik]` is a complex (n, n) matrix.

    `wk` : np.ndarray of shape (nk,)
        Weights associated with each k-point.

    Returns
    -------
    `dos_k` : np.ndarray of shape (ne, nk)
        k-resolved DOS at each energy and k-point:
        dos_k[ie, ik] = -wk[ik] * sum_i Im(G_{ii}(ie, ik)) / π

    `dos` : np.ndarray of shape (ne,)
        Total DOS at each energy, computed as:
        dos[ie] = sum_k dos_k[ie, ik]

    Notes
    -----
    Formula implemented:
        dos_k[ie, ik] = -wk[ik] * Tr[Im G(ie, ik)] / π
        dos[ie] = sum over k of dos_k[ie, ik]

    """
    ne, nk, n, _ = greens_function.shape
    dos_k = np.empty((ne, nk), dtype=np.float64)
    dos = np.zeros(ne, dtype=np.float64)

    for ie in range(ne):
        for ik in range(nk):
            diag_imag = np.imag(np.diagonal(greens_function[ie, ik]))
            dos_k[ie, ik] = -wk[ik] * np.sum(diag_imag) / np.pi
            dos[ie] += dos_k[ie, ik]

    return dos_k, dos
