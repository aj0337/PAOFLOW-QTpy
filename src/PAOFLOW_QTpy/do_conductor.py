from mpi4py import MPI
import numpy as np
from typing import Literal
import numpy.typing as npt


def divide_work(start: int, end: int, rank: int, size: int) -> tuple[int, int]:
    """Divide a 1-indexed range across MPI ranks."""
    total = end - start + 1
    chunk = total // size
    remainder = total % size
    i_start = start + rank * chunk + min(rank, remainder)
    i_end = i_start + chunk - 1
    if rank < remainder:
        i_end += 1
    return i_start, i_end


def run_conductor(
    *,
    ne: int,
    egrid: npt.NDArray[np.float64],
    nkpts_par: int,
    shifts: dict,
    blc_blocks: dict,
    wk_par: npt.NDArray[np.float64],
    conduct_formula: Literal["landauer", "generalized"],
    do_eigenchannels: bool = False,
    neigchnx: int = 0,
    do_eigplot: bool = False,
    ie_eigplot: int | None = None,
    ik_eigplot: int | None = None,
    leads_are_identical: bool = False,
    surface: bool = False,
    lhave_corr: bool = False,
    ldynam_corr: bool = False,
    delta: float = 1e-5,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Main routine for computing the quantum conductance and DOS over energy grid.

    Parameters
    ----------
    `ne` : int
        Number of energy points.
    `egrid` : ndarray
        1D array of energy values.
    `nkpts_par` : int
        Number of parallel k-points.
    `shifts` : dict
        Dictionary with keys 'shift_L', 'shift_C', 'shift_R', 'shift_C_corr'.
    `blc_blocks` : dict
        Dictionary mapping block names to OperatorBlock instances.
    `wk_par` : ndarray
        k-point weights.
    `conduct_formula` : {'landauer', 'generalized'}
        Which conductance formula to use.
    ...

    Returns
    -------
    `conductance` : ndarray
        Total and channel-resolved conductance values vs energy.
    `dos` : ndarray
        Density of states vs energy.
    """

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Step 1: Allocate output arrays
    dimC = blc_blocks["blc_00C"].dim1
    neigchn = (
        min(dimC, blc_blocks["blc_00L"].dim1, blc_blocks["blc_00R"].dim1, neigchnx)
        if do_eigenchannels
        else 0
    )
    conduct = np.zeros((1 + neigchn, ne), dtype=np.float64)
    conduct_k = np.zeros((1 + neigchn, nkpts_par, ne), dtype=np.float64)
    dos = np.zeros(ne, dtype=np.float64)
    dos_k = np.zeros((ne, nkpts_par), dtype=np.float64)

    # Step 2: Determine frequency chunk for this rank
    ie_start, ie_end = divide_work(1, ne, rank, size)

    # Placeholder: Optional file I/O setup can go here

    # Step 3: Loop over energy points (per rank)
    for ie_g in range(ie_start, ie_end + 1):
        # energy = egrid[ie_g - 1]

        # TODO: correlation_read if lhave_corr and ldynam_corr

        for ik in range(nkpts_par):
            # TODO: hamiltonian_setup(ik, ie_g, ie_buff)
            # TODO: build self energies
            # TODO: construct conductor Green's function
            # TODO: compute DOS and transmittance

            pass  # Placeholder for inner k-point loop

        # TODO: communication for DOS and conductance across ranks

    # Step 4: Global reduction
    comm.Allreduce(MPI.IN_PLACE, conduct, op=MPI.SUM)
    comm.Allreduce(MPI.IN_PLACE, conduct_k, op=MPI.SUM)
    comm.Allreduce(MPI.IN_PLACE, dos, op=MPI.SUM)
    comm.Allreduce(MPI.IN_PLACE, dos_k, op=MPI.SUM)

    return conduct, dos
