from __future__ import annotations

import numpy as np

from PAOFLOW_QTpy.operator_blc import OperatorBlock


def initialize_hamiltonian_blocks(
    hk_data: dict[str, np.ndarray],
    nkpts_par: int,
) -> tuple[OperatorBlock, OperatorBlock, OperatorBlock]:
    """
    Initialize Hamiltonian blocks for left lead, conductor, and right lead.

    This function replicates the Fortran logic from `hamiltonian_init.f90`, where the block Hamiltonians
    for the left lead (H00_L), conductor (H00_C), and right lead (H00_R) are read from data sources
    and used to allocate and initialize `OperatorBlock` objects.

    Parameters
    ----------
    `hk_data` : dict
        Dictionary containing Hamiltonian blocks as complex-valued numpy arrays with keys:
        "H00_L", "H00_C", "H00_R". Each array must have shape (dim, dim, nkpts).

    `nkpts_par` : int
        Number of parallel k-points used in the calculation.

    Returns
    -------
    `blc_00L` : OperatorBlock
        Left lead block Hamiltonian operator.

    `blc_00C` : OperatorBlock
        Central (conductor) region block Hamiltonian operator.

    `blc_00R` : OperatorBlock
        Right lead block Hamiltonian operator.

    Notes
    -----

    - `blc_00L%dim1` = size of H00_L block → left lead dimension (dimL)
    - `blc_00C%dim1` = size of H00_C block → conductor dimension (dimC)
    - `blc_00R%dim1` = size of H00_R block → right lead dimension (dimR)

    These values are derived from the input Hamiltonian block shapes.

    Hermiticity is explicitly enforced on H00_L, H00_C, and H00_R,
    consistent with the original Fortran logic using `mat_herm`.
    """
    H00_L = hk_data["H00_L"]
    H00_C = hk_data["H00_C"]
    H00_R = hk_data["H00_R"]

    dimL = H00_L.shape[0]
    dimC = H00_C.shape[0]
    dimR = H00_R.shape[0]

    blc_00L = OperatorBlock(name="blc_00L")
    blc_00L.allocate(dim1=dimL, dim2=dimL, nkpts=nkpts_par)
    blc_00L.H = 0.5 * (H00_L + H00_L.swapaxes(0, 1).conj())

    blc_00C = OperatorBlock(name="blc_00C")
    blc_00C.allocate(dim1=dimC, dim2=dimC, nkpts=nkpts_par)
    blc_00C.H = 0.5 * (H00_C + H00_C.swapaxes(0, 1).conj())

    blc_00R = OperatorBlock(name="blc_00R")
    blc_00R.allocate(dim1=dimR, dim2=dimR, nkpts=nkpts_par)
    blc_00R.H = 0.5 * (H00_R + H00_R.swapaxes(0, 1).conj())

    return blc_00L, blc_00C, blc_00R
