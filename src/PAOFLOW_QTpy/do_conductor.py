from __future__ import annotations

from pathlib import Path
from mpi4py import MPI
import numpy as np
import numpy.typing as npt

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
from PAOFLOW_QTpy.utils.timing import global_timing
from PAOFLOW_QTpy.compute_rham import compute_rham
from PAOFLOW_QTpy.io.get_input_params import ConductorData


def run_conductor(
    data: ConductorData,
    *,
    blc_blocks: dict,
    egrid: npt.NDArray[np.float64],
    wk_par: npt.NDArray[np.float64],
    vkpt_par3D: npt.NDArray[np.float64],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    runtime = data.get_runtime_data()
    dimC = data.dimC
    dimR = data.dimR
    dimL = data.dimL
    ne = data.energy.ne
    delta = data.energy.delta
    conduct_formula = data.conduct_formula
    write_gf = data.symmetry.write_gf
    write_lead_sgm = data.symmetry.write_lead_sgm
    do_eigenchannels = data.symmetry.do_eigenchannels
    neigchnx = data.symmetry.neigchnx
    do_eigplot = data.symmetry.do_eigplot
    ie_eigplot = data.symmetry.ie_eigplot
    ik_eigplot = data.symmetry.ik_eigplot
    leads_are_identical = data.advanced.leads_are_identical
    surface = data.advanced.surface
    niterx = data.iteration.niterx
    transfer_thr = data.iteration.transfer_thr
    nprint = data.iteration.nprint

    ivr_par3D = runtime.ivr_par3D
    vr_par3D = 2 * np.pi * ivr_par3D.astype(np.float64)
    nrtot_par = int(runtime.nrtot_par)
    nkpts_par = int(runtime.nkpts_par)

    neigchn = min(dimC, dimR, dimL, neigchnx) if do_eigenchannels else 0
    conduct = np.zeros((1 + neigchn, ne), dtype=np.float64)
    conduct_k = np.zeros((1 + neigchn, nkpts_par, ne), dtype=np.float64)
    dos = np.zeros(ne, dtype=np.float64)
    dos_k = np.zeros((ne, nkpts_par), dtype=np.float64)

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
                shift_L=data.shift_L,
                shift_C=data.shift_C,
                shift_R=data.shift_R,
                shift_C_corr=getattr(data, "shift_corr", 0.0),
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
                fail_counter=None,
                fail_limit=data.iteration.nfailx,
                verbose=False,
            )

            avg_iter += niter_R + (niter_L if not leads_are_identical else 0)

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
                    transport_dir=data.transport_dir,
                    output_dir=Path("output/eigenchannels"),
                    prefix="eigchn",
                    overwrite=True,
                    verbose=True,
                )

        avg_iter /= 2 * nkpts_par

        if (ie_g % nprint == 0 or ie_g == ie_start or ie_g == ie_end) and rank == 0:
            print(f"  T matrix converged after avg. # of iterations {avg_iter:10.3f}\n")
            global_timing.timing_upto_now(
                "do_conductor", label="Total time spent up to now"
            )

        if write_gf:
            for ir in range(nrtot_par):
                gf_out[ie_g, ir] = compute_rham(
                    vr_par3D[:, ir], gC_k, vkpt_par3D.T, wk_par
                )

        if write_lead_sgm:
            for ir in range(nrtot_par):
                rsgmL_out[ie_g, ir] = compute_rham(
                    vr_par3D[:, ir], sgmL_k, vkpt_par3D.T, wk_par
                )
                rsgmR_out[ie_g, ir] = compute_rham(
                    vr_par3D[:, ir], sgmR_k, vkpt_par3D.T, wk_par
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
            "greenf.xml", gf_out, ivr_par3D, egrid, dimC, True, "retarded", "eV"
        )

    if write_lead_sgm and rank == 0:
        write_operator_xml(
            "lead_L_sgm.xml", rsgmL_out, ivr_par3D, egrid, dimC, True, "retarded", "eV"
        )
        write_operator_xml(
            "lead_R_sgm.xml", rsgmR_out, ivr_par3D, egrid, dimC, True, "retarded", "eV"
        )
        write_kresolved_operator_xml(
            filename=f"lead_L_k.ie{ie_g:04d}.xml",
            operator_k=sgmL_k,
            dimwann=dimC,
            vkpt=vkpt_par3D,
        )

    return conduct, dos, conduct_k, dos_k
