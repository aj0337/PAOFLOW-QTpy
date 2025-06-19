from __future__ import annotations
from pathlib import Path

import numpy as np

from typing import Literal

from PAOFLOW_QTpy.hamiltonian import HamiltonianSystem
from PAOFLOW_QTpy.parsers.read_matrix import read_matrix


def initialize_hamiltonian_blocks(
    ham_system: HamiltonianSystem,
    datafile_C: str,
    ispin: int,
    transport_dir: int,
    calculation_type: Literal["conductor", "bulk"],
    datafile_L: str = "",
    datafile_R: str = "",
    datafile_L_sgm: str = "",
    datafile_R_sgm: str = "",
) -> bool:
    """
    Initialize all Hamiltonian and overlap matrix blocks for the transport system.

    This function allocates and populates the block Hamiltonians and overlaps associated with
    the left lead, right lead, and central conductor. It reads data from `.ham` files generated
    from real-space Hamiltonian blocks, transforms them to k-space, and assigns them to the
    respective `OperatorBlock` instances within the system.

    Depending on the system type ("conductor" or "bulk"), this function loads and links the
    appropriate matrix blocks and applies necessary adjustments. Hermiticity is enforced for
    on-site Hamiltonians and overlaps, and if the left and right leads are structurally identical,
    the left lead blocks are deallocated to avoid redundancy.

    Parameters
    ----------
    `ham_system` : HamiltonianSystem
        Container object holding all OperatorBlock instances for the device.
    `datafile_C` : str
        Path to the `.ham` file for the central conductor region.
    `datafile_L` : str
        Path to the `.ham` file for the left lead region.
    `datafile_R` : str
        Path to the `.ham` file for the right lead region.
    `ispin` : int
        Spin index (0-based) to select the spin channel to load.
    `transport_dir` : int
        Index (1-based) of the transport direction (1 = x, 2 = y, 3 = z).
    `calculation_type` : {"conductor", "bulk"}
        System configuration type. In "bulk" mode, left and right blocks are mirrored from center.
    `datafile_L_sgm` : str, optional
        Path to the self-energy file for the left lead (used for lead identity detection).
    `datafile_R_sgm` : str, optional
        Path to the self-energy file for the right lead (used for lead identity detection).

    Returns
    -------
    `leads_are_identical` : bool
        Whether the left and right lead blocks are structurally and numerically identical.

    Notes
    -----
    This function performs the following key operations:

    - Allocates all relevant Hamiltonian and overlap blocks.
    - Reads the `.ham` data from the specified files into each block.
    - Chooses the appropriate configuration based on `calculation_type`:
        - `"conductor"`: All 7 blocks are populated independently.
        - `"bulk"`: Uses symmetry to copy central blocks into lead blocks.
    - Enforces Hermiticity of the H and S matrices at each k-point:
        H = (H + H†)/2, S = (S + S†)/2
    - Checks whether the left and right leads are identical in:
        - File origin (`datafile_L == datafile_R`)
        - Self-energy files (if provided)
        - Index selection arrays (`irows`, `icols`, `irows_sgm`, `icols_sgm`)
    - If all lead identities match, left blocks are deallocated to save memory.

    The function is designed to work seamlessly with Hamiltonian data read via `read_matrix()`.
    """

    def with_ham_suffix(path: str) -> str:
        return str(Path(path).with_suffix(".ham"))

    ham_system.allocate()

    read_matrix(with_ham_suffix(datafile_C), ispin, transport_dir, ham_system.blc_00C)
    read_matrix(with_ham_suffix(datafile_C), ispin, transport_dir, ham_system.blc_CR)

    if calculation_type == "conductor":
        read_matrix(
            with_ham_suffix(datafile_C), ispin, transport_dir, ham_system.blc_LC
        )
        read_matrix(
            with_ham_suffix(datafile_L), ispin, transport_dir, ham_system.blc_00L
        )
        read_matrix(
            with_ham_suffix(datafile_L), ispin, transport_dir, ham_system.blc_01L
        )
        read_matrix(
            with_ham_suffix(datafile_R), ispin, transport_dir, ham_system.blc_00R
        )
        read_matrix(
            with_ham_suffix(datafile_R), ispin, transport_dir, ham_system.blc_01R
        )

    elif calculation_type == "bulk":
        ham_system.blc_00L = ham_system.blc_00C
        ham_system.blc_00R = ham_system.blc_00C
        ham_system.blc_01L = ham_system.blc_CR
        ham_system.blc_01R = ham_system.blc_CR
        ham_system.blc_LC = ham_system.blc_CR

    else:
        raise ValueError(f"Invalid calculation_type: {calculation_type}")

    nk = ham_system.blc_00C.nkpts
    if nk != ham_system.blc_00L.nkpts or nk != ham_system.blc_00R.nkpts:
        raise RuntimeError("Mismatch in nkpts among C, L, R blocks")

    for ik in range(nk):
        for block in [ham_system.blc_00C, ham_system.blc_00L, ham_system.blc_00R]:
            if block.H is not None:
                H_k = block.H[..., ik]
                block.H[..., ik] = 0.5 * (H_k + H_k.T.conj())
            if block.S is not None:
                S_k = block.S[..., ik]
                block.S[..., ik] = 0.5 * (S_k + S_k.T.conj())

    leads_are_identical = False
    if (
        datafile_L.strip() == datafile_R.strip()
        and datafile_L_sgm.strip() == datafile_R_sgm.strip()
        and all(
            np.array_equal(
                getattr(ham_system.blc_00L, key), getattr(ham_system.blc_00R, key)
            )
            for key in ("irows", "icols", "irows_sgm", "icols_sgm")
        )
        and all(
            np.array_equal(
                getattr(ham_system.blc_01L, key), getattr(ham_system.blc_01R, key)
            )
            for key in ("irows", "icols", "irows_sgm", "icols_sgm")
        )
    ):
        leads_are_identical = True
        ham_system.blc_00L.deallocate()
        ham_system.blc_01L.deallocate()

    return leads_are_identical
