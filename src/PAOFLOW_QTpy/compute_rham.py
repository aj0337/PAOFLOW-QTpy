import numpy as np


def compute_rham(
    rvec: np.ndarray,
    Hk: np.ndarray,
    kpts: np.ndarray,
    wk: np.ndarray,
) -> np.ndarray:
    """
    Compute real-space Hamiltonian H(R) for a single R-vector.

    Parameters
    ----------
    `rvec` : np.ndarray
        Real-space vector R (shape (3,)).
    `Hk` : np.ndarray
        Complex array of shape (nkpts, n, n), the Hamiltonian in k-space.
    `kpts` : np.ndarray
        Array of shape (3, nkpts), the k-point vectors in reciprocal space (cartesian).
    `wk` : np.ndarray
        Array of shape (nkpts,), the integration weights of each k-point.

    Returns
    -------
    `Hr` : np.ndarray
        Complex array of shape (n, n), the Hamiltonian at the given R-vector.

    Notes
    -----
    The function implements the inverse Fourier transform of the Hamiltonian
    from reciprocal space to real space using the formula:

        H(R) = ∑_k w_k · exp(-i k · R) · H(k)

    where:
        - `R` is a lattice vector in real space (in Cartesian units),
        - `k` is a vector in reciprocal space (Cartesian),
        - `w_k` is the integration weight associated with `k`,
        - `H(k)` is the Hamiltonian in reciprocal space.

    """
    phase = np.exp(-1j * np.dot(kpts.T, rvec))
    Hr = np.einsum("k,kmn->mn", phase * wk, Hk)
    return Hr
