from __future__ import annotations

from pathlib import Path
from typing import Literal
from mpi4py import MPI

import numpy as np
import numpy.typing as npt
import time

from PAOFLOW_QTpy.io.write_data import write_eigenchannels, write_kresolved_data
from PAOFLOW_QTpy.green import compute_conductor_green_function
from PAOFLOW_QTpy.hamiltonian_setup import hamiltonian_setup
from PAOFLOW_QTpy.leads_self_energy import build_self_energies_from_blocks
from PAOFLOW_QTpy.transmittance import evaluate_transmittance
from PAOFLOW_QTpy.utils.divide_et_impera import divide_work


def run_conductor(
    *,
    data_dict: dict,
    ne: int,
    egrid: npt.NDArray[np.float64],
    nkpts_par: int,
    shifts: dict,
    blc_blocks: dict,
    wk_par: npt.NDArray[np.float64],
    vkpt_par3D: npt.NDArray[np.float64],
    transport_dir: int,
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
    niterx: int = 200,
    transfer_thr: float = 1e-12,
    fail_counter: dict | None = None,
    fail_limit: int = 10,
    verbose: bool = False,
    nprint: int = 20,
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

    dimC = blc_blocks["blc_00C"].dim1
    dimR = blc_blocks["blc_00R"].dim1

    neigchn = min(dimC, dimR, dimC, neigchnx) if do_eigenchannels else 0

    conduct = np.zeros((1 + neigchn, ne), dtype=np.float64)
    conduct_k = np.zeros((1 + neigchn, nkpts_par, ne), dtype=np.float64)
    dos = np.zeros(ne, dtype=np.float64)
    dos_k = np.zeros((ne, nkpts_par), dtype=np.float64)

    ie_start, ie_end = divide_work(1, ne, rank, size)

    for ie_g in range(ie_start, ie_end + 1):
        avg_iter = 0.0

        if (ie_g % nprint == 0 or ie_g == ie_start or ie_g == ie_end) and rank == 0:
            print(f"  Computing E({ie_g:6d}) = {egrid[ie_g]:12.5f} eV")

        for ik in range(nkpts_par):
            hamiltonian_setup(
                ik=ik,
                ie_g=ie_g,
                egrid=egrid,
                shift_L=shifts["shift_L"],
                shift_C=shifts["shift_C"],
                shift_R=shifts["shift_R"],
                shift_C_corr=shifts.get("shift_C_corr", 0.0),
                blc_blocks=blc_blocks,
                ie_buff=1,
            )

            sigma_R, sigma_L, niter_R, niter_L = build_self_energies_from_blocks(
                blc_00R=blc_blocks["blc_00R"].aux,
                blc_01R=blc_blocks["blc_01R"].aux,
                blc_00L=blc_blocks["blc_00L"].aux,
                blc_01L=blc_blocks["blc_01L"].aux,
                blc_CR=blc_blocks["blc_CR"].aux,
                blc_LC=blc_blocks["blc_LC"].aux,
                leads_are_identical=leads_are_identical,
                delta=delta,
                niterx=niterx,
                transfer_thr=transfer_thr,
                fail_counter=fail_counter,
                fail_limit=fail_limit,
                verbose=verbose,
            )

            avg_iter += niter_R + (niter_R if leads_are_identical else niter_L)

            gC = compute_conductor_green_function(
                energy=egrid[ie_g],
                h_c=blc_blocks["blc_00C"].H[..., ik],
                s_c=blc_blocks["blc_00C"].S[..., ik],
                sigma_l=sigma_L,
                sigma_r=sigma_R if not surface else None,
                delta=delta,
                surface=surface,
            )

            diag_imag = np.imag(np.diagonal(gC))
            dos_k[ie_g, ik] = -wk_par[ik] * np.sum(diag_imag) / np.pi
            dos[ie_g] += dos_k[ie_g, ik]

            gamma_L = 1j * (sigma_L - sigma_L.conj().T)
            gamma_R = 1j * (sigma_R - sigma_R.conj().T)

            cond_aux, z_eigplot = evaluate_transmittance(
                gamma_L=gamma_L,
                gamma_R=gamma_R,
                G_ret=gC,
                formula=conduct_formula,
                do_eigenchannels=do_eigenchannels,
                do_eigplot=(
                    do_eigenchannels
                    and do_eigplot
                    and ie_g == ie_eigplot
                    and ik == ik_eigplot
                ),
                sgm_corr=None,
                eta=delta,
                S_overlap=None,
            )

            conduct[0, ie_g] += wk_par[ik] * np.sum(cond_aux)
            conduct_k[0, ik, ie_g] += wk_par[ik] * np.sum(cond_aux)

            if do_eigenchannels:
                nchan = min(neigchn, cond_aux.shape[0])
                conduct[1 : 1 + nchan, ie_g] += wk_par[ik] * cond_aux[:nchan]
                conduct_k[1 : 1 + nchan, ik, ie_g] += wk_par[ik] * cond_aux[:nchan]

            if (
                do_eigenchannels
                and do_eigplot
                and z_eigplot is not None
                and ie_g == ie_eigplot
                and ik == ik_eigplot
                and rank == 0
            ):
                write_eigenchannels(
                    data=z_eigplot,
                    ie=ie_g,
                    ik=ik,
                    vkpt=vkpt_par3D[:, ik],
                    transport_dir=transport_dir,
                    output_dir=Path("output/eigenchannels"),
                    prefix="eigchn",
                    overwrite=True,
                    verbose=True,
                )

        avg_iter /= 2 * nkpts_par

        if (ie_g % nprint == 0 or ie_g == ie_start or ie_g == ie_end) and rank == 0:
            print(f"  T matrix converged after avg. # of iterations {avg_iter:10.3f}")
            elapsed = time.perf_counter() - data_dict["_freqloop_start_time"]
            print(f"\n{'Total time spent up to now :':>40} {elapsed:10.2f} secs\n")

    comm.Allreduce(MPI.IN_PLACE, conduct, op=MPI.SUM)
    comm.Allreduce(MPI.IN_PLACE, conduct_k, op=MPI.SUM)
    comm.Allreduce(MPI.IN_PLACE, dos, op=MPI.SUM)
    comm.Allreduce(MPI.IN_PLACE, dos_k, op=MPI.SUM)

    if rank == 0:
        output_dir = Path("output")
        write_kresolved_data(
            egrid,
            conduct_k,
            label="cond",
            output_dir=output_dir,
            prefix="prefix",
            postfix="",
        )
        write_kresolved_data(
            egrid,
            dos_k,
            label="doscond",
            output_dir=output_dir,
            prefix="prefix",
            postfix="",
        )

    return conduct, dos
