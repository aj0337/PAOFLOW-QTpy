from __future__ import annotations

from pathlib import Path
from typing import Literal
from mpi4py import MPI
import numpy as np
import numpy.typing as npt
import time

from PAOFLOW_QTpy.io.write_data import (
    write_eigenchannels,
    write_operator_xml,
    write_kresolved_operator_xml,
)
from PAOFLOW_QTpy.green import compute_conductor_green_function
from PAOFLOW_QTpy.hamiltonian_setup import hamiltonian_setup
from PAOFLOW_QTpy.leads_self_energy import build_self_energies_from_blocks
from PAOFLOW_QTpy.transmittance import evaluate_transmittance
from PAOFLOW_QTpy.utils.divide_et_impera import divide_work
from PAOFLOW_QTpy.compute_rham import compute_rham


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
    write_gf: bool = True,  # TODO Make variable an input from yaml file
    gf_filename: str = "greenf.xml",
    write_lead_sgm: bool = True,  # TODO Make variable an input from yaml file
    lead_sgm_prefix: str = "lead",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute conductance and DOS over an energy grid, write optional operator data.

    Parameters
    ----------
    `data_dict` : dict
        Must provide ``nrtot_par``, ``ivr_par3D``, ``vr_par3D``, and ``_freqloop_start_time``.
    `ne` : int
        Number of energy points.
    `egrid` : ndarray
        Energy grid of shape ``(ne,)`` in eV.
    `nkpts_par` : int
        Number of parallel k-points.
    `shifts` : dict
        Contains ``shift_L``, ``shift_C``, ``shift_R``, and optionally ``shift_C_corr``.
    `blc_blocks` : dict
        Mapping of block names to OperatorBlock instances: ``blc_00L``, ``blc_01L``, ``blc_00R``,
        ``blc_01R``, ``blc_00C``, ``blc_LC``, ``blc_CR``.
    `wk_par` : ndarray
        k-point weights of shape ``(nkpts_par,)``.
    `vkpt_par3D` : ndarray
        k-points in Cartesian coordinates of shape ``(3, nkpts_par)``.
    `transport_dir` : int
        Transport direction index.
    `conduct_formula` : {'landauer', 'generalized'}
        Choice of transmittance formula.
    `do_eigenchannels` : bool
        Whether to compute eigenchannel-resolved conductance.
    `neigchnx` : int
        Maximum number of eigenchannels to keep.
    `do_eigplot` : bool
        Whether to write eigenchannel auxiliary data for a specific ``(ie, ik)``.
    `ie_eigplot` : int or None
        Energy index for eigenchannel snapshot.
    `ik_eigplot` : int or None
        k-point index for eigenchannel snapshot.
    `leads_are_identical` : bool
        Reuse right-lead solution for left lead if True.
    `surface` : bool
        Compute projected surface Green's function if True.
    `lhave_corr` : bool
        Placeholder for correlated calculations.
    `ldynam_corr` : bool
        Placeholder for dynamical correlations.
    `delta` : float
        Positive infinitesimal for Green's function.
    `niterx` : int
        Maximum iterations for surface transfer matrix solver.
    `transfer_thr` : float
        Convergence threshold for transfer matrices.
    `fail_counter` : dict or None
        Optional failure tracking dictionary.
    `fail_limit` : int
        Maximum consecutive failures allowed before aborting.
    `verbose` : bool
        Verbose iteration logs for solvers.
    `nprint` : int
        Frequency of progress prints.
    `write_gf` : bool
        Write conductor Green's function to disk if True.
    `gf_filename` : str
        Output filename for conductor Green's function.
    `write_lead_sgm` : bool
        Write lead self-energies to disk if True.
    `lead_sgm_prefix` : str
        Prefix for lead self-energy output filenames.

    Returns
    -------
    `conduct` : ndarray
        Array of shape ``(1 + neigchn, ne)`` with total and channel-resolved conductance.
    `dos` : ndarray
        Array of shape ``(ne,)`` with DOS of the conductor.
    `conduct_k` : ndarray
        Array of shape ``(1 + neigchn, nkpts_par, ne)`` with k-resolved conductance.
    `dos_k` : ndarray
        Array of shape ``(ne, nkpts_par)`` with k-resolved DOS.

    Notes
    -----
    The real-space operators written to disk are built via the inverse Bloch transform

    .. math::

        O(R) = \\sum_{k} w_k\\,e^{-i k\\cdot R}\\,O(k)

    where ``O`` is any operator defined at each k-point (e.g., the retarded Green’s
    function of the conductor ``G^r`` or the lead self-energies ``\\Sigma_{L,R}``),
    ``w_k`` are the integration weights, and ``R`` are the real-space lattice vectors
    matching ``ivr_par3D``/``vr_par3D``.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    dimC = blc_blocks["blc_00C"].dim1
    dimR = blc_blocks["blc_00R"].dim1
    dimL = blc_blocks["blc_00L"].dim1
    neigchn = min(dimC, dimR, dimL, neigchnx) if do_eigenchannels else 0

    conduct = np.zeros((1 + neigchn, ne), dtype=np.float64)
    conduct_k = np.zeros((1 + neigchn, nkpts_par, ne), dtype=np.float64)
    dos = np.zeros(ne, dtype=np.float64)
    dos_k = np.zeros((ne, nkpts_par), dtype=np.float64)

    nrtot_par = int(data_dict["nrtot_par"])
    ivr_par3D = data_dict["ivr_par3D"]
    vr_par3D = 2 * np.pi * ivr_par3D.astype(np.float64)

    gf_out = (
        np.zeros((ne, nrtot_par, dimC, dimC), dtype=np.complex128) if write_gf else None
    )
    rsgmL_out = (
        np.zeros((ne, nrtot_par, dimC, dimC), dtype=np.complex128)
        if write_lead_sgm
        else None
    )
    rsgmR_out = (
        np.zeros((ne, nrtot_par, dimC, dimC), dtype=np.complex128)
        if write_lead_sgm
        else None
    )

    ie_start, ie_end = divide_work(0, ne - 1, rank, size)

    for ie_g in range(ie_start, ie_end + 1):
        avg_iter = 0.0

        if (ie_g % nprint == 0 or ie_g == ie_start or ie_g == ie_end) and rank == 0:
            print(f"  Computing E({ie_g:6d}) = {egrid[ie_g]:12.5f} eV")

        gC_k = (
            np.zeros((nkpts_par, dimC, dimC), dtype=np.complex128) if write_gf else None
        )
        sgmL_k = (
            np.zeros((nkpts_par, dimC, dimC), dtype=np.complex128)
            if write_lead_sgm
            else None
        )
        sgmR_k = (
            np.zeros((nkpts_par, dimC, dimC), dtype=np.complex128)
            if write_lead_sgm
            else None
        )

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
                blc_00R=blc_blocks["blc_00R"].at_k(ik),
                blc_01R=blc_blocks["blc_01R"].at_k(ik),
                blc_00L=blc_blocks["blc_00L"].at_k(ik),
                blc_01L=blc_blocks["blc_01L"].at_k(ik),
                blc_CR=blc_blocks["blc_CR"].at_k(ik),
                blc_LC=blc_blocks["blc_LC"].at_k(ik),
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
                blc_00C=blc_blocks["blc_00C"].at_k(ik),
                sigma_l=sigma_L,
                sigma_r=sigma_R if not surface else None,
                delta=delta,
                surface=surface,
            )

            diag_imag = np.imag(np.diagonal(gC))
            dos_k[ie_g, ik] = -wk_par[ik] * np.sum(diag_imag) / np.pi
            dos[ie_g] += dos_k[ie_g, ik]

            if write_gf:
                gC_k[ik] = gC
            if write_lead_sgm:
                sgmL_k[ik] = sigma_L
                sgmR_k[ik] = sigma_R

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

        if write_gf:
            for ir in range(nrtot_par):
                gf_out[ie_g, ir] = compute_rham(
                    rvec=vr_par3D[:, ir],
                    Hk=gC_k,
                    kpts=vkpt_par3D.T,
                    wk=wk_par,
                )

        if write_lead_sgm:
            for ir in range(nrtot_par):
                rsgmL_out[ie_g, ir] = compute_rham(
                    rvec=vr_par3D[:, ir],
                    Hk=sgmL_k,
                    kpts=vkpt_par3D.T,
                    wk=wk_par,
                )
                rsgmR_out[ie_g, ir] = compute_rham(
                    rvec=vr_par3D[:, ir],
                    Hk=sgmR_k,
                    kpts=vkpt_par3D.T,
                    wk=wk_par,
                )

    comm.Allreduce(MPI.IN_PLACE, conduct, op=MPI.SUM)
    comm.Allreduce(MPI.IN_PLACE, conduct_k, op=MPI.SUM)
    comm.Allreduce(MPI.IN_PLACE, dos, op=MPI.SUM)
    comm.Allreduce(MPI.IN_PLACE, dos_k, op=MPI.SUM)

    if write_gf:
        comm.Allreduce(MPI.IN_PLACE, gf_out, op=MPI.SUM)
    if write_lead_sgm:
        comm.Allreduce(MPI.IN_PLACE, rsgmL_out, op=MPI.SUM)
        comm.Allreduce(MPI.IN_PLACE, rsgmR_out, op=MPI.SUM)

    if write_gf and rank == 0:
        write_operator_xml(
            filename=gf_filename,
            operator_matrix=gf_out,
            ivr=ivr_par3D,
            grid=egrid,
            dimwann=dimC,
            dynamical=True,
            analyticity="retarded",
            eunits="eV",
        )

    if write_lead_sgm and rank == 0:
        write_operator_xml(
            filename=f"{lead_sgm_prefix}_L_sgm.xml",
            operator_matrix=rsgmL_out,
            ivr=ivr_par3D,
            grid=egrid,
            dimwann=dimC,
            dynamical=True,
            analyticity="retarded",
            eunits="eV",
        )
        write_operator_xml(
            filename=f"{lead_sgm_prefix}_R_sgm.xml",
            operator_matrix=rsgmR_out,
            ivr=ivr_par3D,
            grid=egrid,
            dimwann=dimC,
            dynamical=True,
            analyticity="retarded",
            eunits="eV",
        )
        outname = f"{lead_sgm_prefix}_L_k.ie{ie_g:04d}.xml"
        write_kresolved_operator_xml(
            filename=outname,
            operator_k=sgmL_k,
            dimwann=dimC,
            vkpt=vkpt_par3D,
        )
    return conduct, dos, conduct_k, dos_k
