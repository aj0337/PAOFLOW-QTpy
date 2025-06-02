import numpy as np
from typing import Tuple


def kpoints_rmask(rvect: np.ndarray, transport_dir: int) -> np.ndarray:
    """
    Expand a 2D reciprocal vector `rvect` to 3D according to the transport direction.

    Parameters
    ----------
    `rvect` : (2,) ndarray
        2D k-point or R-vector in reciprocal lattice units (fractional coordinates).
    `transport_dir` : int
        Direction of transport (1 for x, 2 for y, 3 for z).

    Returns
    -------
    `rmask` : (3,) ndarray
        Expanded 3D vector in reciprocal lattice units, with zeros in the transport direction.

    Notes
    -----
    This function maps a 2D vector in the plane perpendicular to the transport direction
    into a full 3D vector by inserting zeros along the transport direction.
    For example, if the transport direction is z (3), a 2D k-point (kx, ky) is mapped to (kx, ky, 0).

    Mathematically:
        Let `rvect` = (r₁, r₂)
        The 3D expansion is:
            if transport_dir == 1 (x):   (0, r₁, r₂)
            if transport_dir == 2 (y):   (r₁, 0, r₂)
            if transport_dir == 3 (z):   (r₁, r₂, 0)
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
    Check if two 2D k-points are equivalent under time-reversal symmetry (modulo 1).

    Parameters
    ----------
    `v1`, `v2` : (2,) ndarray
        2D k-point vectors in reciprocal lattice units (fractional coordinates).
    `tol` : float
        Tolerance for equality check.

    Returns
    -------
    `is_equiv` : bool
        True if v1 ≈ -v2 (mod 1), i.e., they are time-reversal partners.

    Notes
    -----
    In a time-reversal symmetric system, a k-point `k` is equivalent to `-k` (modulo the reciprocal lattice).
    This function checks:
        (v1 + v2) % 1 ≈ 0

    For example:
        v1 = (0.25, 0.5), v2 = (-0.25, -0.5) -> equivalent
        v1 = (0.25, 0.5), v2 = (0.25, 0.5)   -> not equivalent (unless v1 == 0)

    This is essential for symmetrizing the k-point mesh and avoiding double counting.
    """
    return np.allclose((v1 + v2) % 1.0, 0.0, atol=tol)


def initialize_kpoints(
    nk_par: Tuple[int, int],
    s_par: Tuple[int, int],
    transport_dir: int,
    use_symm: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate 3D k-points and weights on a uniform 2D mesh orthogonal to the transport direction.

    Parameters
    ----------
    `nk_par` : Tuple[int, int]
        Number of k-points along the two non-transport directions.
    `s_par` : Tuple[int, int]
        Shifts (in fractional units) for the mesh in the two non-transport directions.
    `transport_dir` : int
        Transport direction (1 = x, 2 = y, 3 = z).
    `use_symm` : bool
        Whether to symmetrize the mesh under time-reversal (k ≡ -k).

    Returns
    -------
    `vkpt_par3D` : (nkpts, 3) ndarray
        Generated 3D k-points in fractional coordinates.
    `wk_par` : (nkpts,) ndarray
        Normalized weights for each k-point.

    Notes
    -----
    The k-point grid is generated as:
        kx(i) = (i - nkx//2) / nkx + shift_x / (2 * nkx)
        ky(j) = (j - nky//2) / nky + shift_y / (2 * nky)
    for i in [0, nkx-1], j in [0, nky-1].

    If `use_symm` is True, pairs of points (k, -k) are considered equivalent under time-reversal symmetry
    and only unique representatives are kept.

    The resulting weights are normalized such that:
        sum(wk_par) = 1.

    Mathematically:
        For a uniform mesh in the 2D plane orthogonal to transport:
            k = (kx, ky, 0) for transport_dir = 3
        where the 3D extension is given by `kpoints_rmask`.

    This is analogous to Monkhorst-Pack grids used in bandstructure calculations.
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
    Compute Fourier phase factors exp(i 2π k · R) for each pair of k-point and R-vector.

    Parameters
    ----------
    `vkpts` : (nkpts, 2) or (nkpts, 3) ndarray
        k-point vectors in fractional reciprocal lattice units.
    `ivr_par` : (nR, 2) or (nR, 3) ndarray
        R-vectors in fractional crystal units (integer multiples of lattice vectors).

    Returns
    -------
    `table_par` : (nR, nkpts) ndarray of complex128
        Phase factors e^{i 2π k · R} used for Fourier transforms or interpolation.

    Notes
    -----
    This computes the plane-wave phase factors:
        table_par[ir, ik] = exp(i * 2π * (k · R))
    where:
        k : reciprocal vector (fractional units)
        R : real-space vector (fractional units)

    These factors are essential for transforming quantities between real and reciprocal space.
    For example, they are used in:
        - Evaluating Fourier series expansions
        - Computing Green's functions or self-energies in k-space
        - Constructing Hamiltonians or overlaps in different representations

    The 2π factor comes from the convention of expressing k and R in fractional units:
        k = k_cartesian / (2π)  ->  k_cartesian = 2π * k_fractional
    """
    nR = ivr_par.shape[0]
    nkpts = vkpts.shape[0]

    table = np.empty((nR, nkpts), dtype=np.complex128)
    for ik in range(nkpts):
        for ir in range(nR):
            arg = 2 * np.pi * np.dot(vkpts[ik], ivr_par[ir])
            table[ir, ik] = np.exp(1j * arg)
    return table


def initialize_r_vectors(
    nr_par: Tuple[int, int],
    transport_dir: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate 3D R-vectors and weights based on a uniform 2D integer grid orthogonal to the transport direction.

    Parameters
    ----------
    `nr_par` : Tuple[int, int]
        2D mesh sizes for R-vectors in the directions orthogonal to transport.
    `transport_dir` : int
        Direction of transport (1-based, 1 = x, 2 = y, 3 = z).

    Returns
    -------
    `ivr_par3D` : (nR, 3) ndarray of int
        3D integer R-vectors in crystal coordinates.
    `wr_par` : (nR,) ndarray of float
        Weights for each R-vector (normalized to match Fortran behavior).

    Notes
    -----
    The 2D R-vectors are generated as:
        R_i = i - (nr_x + 1) // 2
        R_j = j - (nr_y + 1) // 2
    for i in [1, nr_x], j in [1, nr_y].

    Hermitian symmetry is enforced by ensuring that for each R, -R is present.
    If a corresponding -R is not found, it is added, and the weights of both R and -R are halved.

    The 2D vectors are then expanded to 3D using:
        if transport_dir == 1: (0, R1, R2)
        if transport_dir == 2: (R1, 0, R2)
        if transport_dir == 3: (R1, R2, 0)
    """

    nx, ny = nr_par
    R_list = []
    w_list = []

    for j in range(1, ny + 1):
        for i in range(1, nx + 1):
            Rx = i - (nx + 1) // 2
            Ry = j - (ny + 1) // 2
            R_list.append([Rx, Ry])
            w_list.append(1.0)

    R_array = np.array(R_list, dtype=int)
    w_array = np.array(w_list, dtype=np.float64)

    counter = len(R_array)
    i = 0
    while i < counter:
        R = R_array[i]
        found = np.any(np.all(R_array[:counter] == -R, axis=1))
        if not found:
            R_array = np.vstack([R_array, -R])
            w_array = np.append(w_array, 0.5 * w_array[i])
            w_array[i] *= 0.5
            counter += 1
        i += 1

    def kpoints_imask(ivect: np.ndarray, transport_dir: int) -> np.ndarray:
        imask = np.zeros(3, dtype=int)
        if transport_dir == 1:
            imask[1:] = ivect
        elif transport_dir == 2:
            imask[0] = ivect[0]
            imask[2] = ivect[1]
        elif transport_dir == 3:
            imask[:2] = ivect
        else:
            raise ValueError(f"Invalid transport direction: {transport_dir}")
        return imask

    ivr_par3D = np.array([kpoints_imask(R, transport_dir) for R in R_array])
    wr_par = w_array

    return ivr_par3D, wr_par
