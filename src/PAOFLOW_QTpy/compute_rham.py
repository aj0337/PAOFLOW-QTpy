import numpy as np


def compute_rham(
    Hk: np.ndarray,
    kpts: np.ndarray,
    wk: np.ndarray,
    rvecs: np.ndarray,
) -> np.ndarray:
    """
    Compute real-space Hamiltonians H(R) from k-space Hamiltonians H(k)
    via inverse Fourier transform over the Brillouin zone.

    Parameters
    ----------
    `Hk` : np.ndarray
        Complex array of shape (nspin, nkpts, n, n), the Hamiltonian in k-space.
    `kpts` : np.ndarray
        Array of shape (3, nkpts), the k-point vectors in reciprocal space (cartesian).
    `wk` : np.ndarray
        Array of shape (nkpts,), the integration weights of each k-point.
    `rvecs` : np.ndarray
        Array of shape (nR, 3), the real-space R-vectors in direct lattice coordinates (cartesian).

    Returns
    -------
    `Hr` : np.ndarray
        Complex array of shape (nspin, nR, n, n), real-space Hamiltonian H(R).

    Notes
    -----
    This function implements the inverse Fourier transform of the Hamiltonian
    from reciprocal space to real space using the formula:

        H(R) = ∑_k w_k · exp(-i k · R) · H(k)

    where:
        - R is a lattice vector in real space (in Cartesian units),
        - k is a vector in reciprocal space (also Cartesian),
        - w_k is the integration weight associated with k,
        - H(k) is the Hamiltonian in reciprocal space.

    The scalar product k · R must be dimensionless, hence both vectors must
    be expressed in consistent units (e.g., Cartesian).


    """
    nspin, nkpts, n, _ = Hk.shape
    nR = rvecs.shape[0]

    Hr = np.zeros((nspin, nR, n, n), dtype=np.complex128)

    phase_factors = np.exp(
        -1j * np.einsum("rk,kp->rp", rvecs, kpts) * wk[np.newaxis, :]
    )

    for isp in range(nspin):
        Hr[isp] = np.einsum("rp,pmn->rmn", phase_factors, Hk[isp])

    return Hr
