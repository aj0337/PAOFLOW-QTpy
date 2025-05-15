import numpy as np
from typing import Tuple


def kpoints_rmask(rvect: np.ndarray, transport_dir: int) -> np.ndarray:
    """
    Expand 2D reciprocal vector `rvect` to 3D according to transport direction.

    Parameters
    ----------
    `rvect` : (2,) ndarray
        2D k-point or R-vector.
    `transport_dir` : int
        Direction of transport (1, 2, or 3).

    Returns
    -------
    `rmask` : (3,) ndarray
        Expanded 3D vector in crystal coordinates.
    """
    rmask = np.zeros(3)
    if transport_dir == 1:
        rmask[1:] = rvect
    elif transport_dir == 2:
        rmask[0] = rvect[0]
        rmask[2] = rvect[1]
    elif transport_dir == 3:
        rmask[:2] = rvect
    else:
        raise ValueError(f"Invalid transport direction: {transport_dir}")
    return rmask


def kpoints_equivalent(v1: np.ndarray, v2: np.ndarray, tol: float = 1e-6) -> bool:
    """
    Check if two 2D k-points are equivalent under time-reversal symmetry.

    Parameters
    ----------
    `v1`, `v2` : (2,) ndarray
        k-point vectors.
    `tol` : float
        Tolerance for modulo check.

    Returns
    -------
    `is_equiv` : bool
        True if v1 ≈ -v2 (mod 1)
    """
    return np.allclose((v1 + v2) % 1.0, 0.0, atol=tol)


def initialize_kpoints(
    nk_par: Tuple[int, int],
    s_par: Tuple[int, int],
    transport_dir: int,
    use_symm: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate 3D k-points and weights based on a 2D mesh orthogonal to the transport direction.

    Parameters
    ----------
    `nk_par` : Tuple[int, int]
        2D mesh along the two directions orthogonal to transport.
    `s_par` : Tuple[int, int]
        2D shifts in those directions.
    `transport_dir` : int
        Direction of transport (1-based, 1 = x, 2 = y, 3 = z).
    `use_symm` : bool
        Whether to enforce k ≡ -k symmetry.

    Returns
    -------
    `vkpt_par3D` : ndarray of shape (nkpts, 3)
        3D k-point vectors in crystal units.
    `wk_par` : ndarray of shape (nkpts,)
        k-point weights (normalized).
    """
    mesh_x, mesh_y = nk_par
    shift_x, shift_y = s_par
    vkpts_2d = []
    weights = []

    for j in range(mesh_y):
        for i in range(mesh_x):
            kx = (i - mesh_x // 2) / mesh_x + shift_x / (2 * mesh_x)
            ky = (j - mesh_y // 2) / mesh_y + shift_y / (2 * mesh_y)
            kpt = np.array([kx, ky])
            if use_symm:
                for existing in vkpts_2d:
                    if kpoints_equivalent(existing, kpt):
                        break
                else:
                    vkpts_2d.append(kpt)
                    weights.append(1.0)
            else:
                vkpts_2d.append(kpt)
                weights.append(1.0)

    vkpts_2d = np.array(vkpts_2d)
    wk_par = np.array(weights)
    wk_par /= wk_par.sum()

    vkpt_par3D = np.array([kpoints_rmask(kpt, transport_dir) for kpt in vkpts_2d])
    return vkpt_par3D, wk_par


def compute_fourier_phase_table(
    vkpts: np.ndarray,
    ivr_par: np.ndarray,
) -> np.ndarray:
    """
    Compute Fourier transform phase factors exp(i 2π k · R) for FFT table.

    Parameters
    ----------
    `vkpts` : (nkpts, 2) or (nkpts, 3) ndarray
        k-point vectors in crystal units.
    `ivr_par` : (nR, 2) or (nR, 3) ndarray
        R-vectors in crystal units (integer format).

    Returns
    -------
    `table_par` : (nR, nkpts) ndarray of complex128
        Phase factors e^{i 2π k·R} for FFT or interpolation.
    """
    nR = ivr_par.shape[0]
    nkpts = vkpts.shape[0]

    table = np.empty((nR, nkpts), dtype=np.complex128)
    for ik in range(nkpts):
        for ir in range(nR):
            arg = 2 * np.pi * np.dot(vkpts[ik], ivr_par[ir])
            table[ir, ik] = np.exp(1j * arg)
    return table
