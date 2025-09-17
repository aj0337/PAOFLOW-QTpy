from __future__ import annotations

from pathlib import Path
from mpi4py import MPI
import numpy as np
import os

from PAOFLOW_QTpy.io.write_data import (
    write_eigenchannels,
    write_operator_xml,
    write_kresolved_operator_xml,
)
from PAOFLOW_QTpy.transport.green import compute_conductor_green_function
from PAOFLOW_QTpy.hamiltonian.hamiltonian_setup import hamiltonian_setup
from PAOFLOW_QTpy.io.write_header import headered_function
from PAOFLOW_QTpy.transport.leads_self_energy import build_self_energies_from_blocks
from PAOFLOW_QTpy.transport.transmittance import evaluate_transmittance
from PAOFLOW_QTpy.utils.divide_et_impera import divide_work
from PAOFLOW_QTpy.utils.timing import global_timing, timed_function
from PAOFLOW_QTpy.hamiltonian.compute_rham import compute_rham
from PAOFLOW_QTpy.io.get_input_params import ConductorData
from PAOFLOW_QTpy.io.write_data import write_data


class ConductorCalculator:
    def __init__(
        self,
        data: ConductorData,
        *,
        blc_blocks: dict,
    ):
        self.data = data
        self.blc_blocks = blc_blocks
        self.vkpt_par3D = data._runtime.vkpt_par3D

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        self.runtime = data.get_runtime_data()
        self.dimC = data.dimC
        self.dimR = data.dimR
        self.dimL = data.dimL
        self.ne = data.energy.ne
        self.delta = data.energy.delta
        self.nkpts_par = int(self.runtime.nkpts_par)
        self.egrid = np.linspace(data.energy.emin, data.energy.emax, data.energy.ne)
        self.wk_par = data._runtime.wk_par

        self.ivr_par3D = self.runtime.ivr_par3D
        self.vr_par3D = 2 * np.pi * self.ivr_par3D.astype(np.float64)
        self.nrtot_par = int(self.runtime.nrtot_par)

    @timed_function("do_conductor")
    @headered_function("Frequency Loop")
    def run(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self.conduct, self.dos, self.conduct_k, self.dos_k = self.initialize_outputs()
        ie_start, ie_end = divide_work(0, self.ne - 1, self.rank, self.size)
        for ie_g in range(ie_start, ie_end + 1):
            self.conduct, self.dos = self.process_energy(
                self.conduct, self.dos, self.conduct_k, self.dos_k, ie_g
            )
        self.reduce_results(self.conduct, self.dos, self.conduct_k, self.dos_k)
        self.write_operators()

        return self.conduct, self.dos, self.conduct_k, self.dos_k

    def initialize_outputs(self):
        do_eigenchannels = self.data.symmetry.do_eigenchannels
        neigchnx = self.data.symmetry.neigchnx
        neigchn = (
            min(self.dimC, self.dimR, self.dimL, neigchnx) if do_eigenchannels else 0
        )

        conduct = np.zeros((1 + neigchn, self.ne), dtype=np.float64)
        conduct_k = np.zeros((1 + neigchn, self.nkpts_par, self.ne), dtype=np.float64)
        dos = np.zeros(self.ne, dtype=np.float64)
        dos_k = np.zeros((self.ne, self.nkpts_par), dtype=np.float64)

        self.gf_out = (
            np.zeros(
                (self.ne, self.nrtot_par, self.dimC, self.dimC), dtype=np.complex128
            )
            if self.data.symmetry.write_gf
            else None
        )
        self.rsgmL_out = (
            np.zeros(
                (self.ne, self.nrtot_par, self.dimC, self.dimC), dtype=np.complex128
            )
            if self.data.symmetry.write_lead_sgm
            else None
        )
        self.rsgmR_out = (
            np.zeros(
                (self.ne, self.nrtot_par, self.dimC, self.dimC), dtype=np.complex128
            )
            if self.data.symmetry.write_lead_sgm
            else None
        )

        return conduct, dos, conduct_k, dos_k

    def process_energy(self, conduct, dos, conduct_k, dos_k, ie_g: int):
        nprint = self.data.iteration.nprint
        if (ie_g % nprint == 0 or ie_g == 0 or ie_g == self.ne - 1) and self.rank == 0:
            print(f"  Computing E({ie_g:6d}) = {self.egrid[ie_g]:12.5f} eV")

        gC_k, sgmL_k, sgmR_k = self.initialize_k_dependent()
        avg_iter = 0.0

        for ik in range(self.nkpts_par):
            gC, sigma_L, sigma_R, niter_sum = self.process_kpoint(ie_g, ik)
            avg_iter += niter_sum

            self.accumulate_dos(dos, dos_k, gC, ie_g, ik)
            self.accumulate_conductance(
                conduct, conduct_k, gC, sigma_L, sigma_R, ie_g, ik
            )

            if self.data.symmetry.write_gf:
                gC_k[ik] = gC
            if self.data.symmetry.write_lead_sgm:
                sgmL_k[ik], sgmR_k[ik] = sigma_L, sigma_R

        self.finalize_energy(avg_iter, ie_g, gC_k, sgmL_k, sgmR_k)
        return conduct, dos

    def initialize_k_dependent(self):
        gC_k = (
            np.zeros((self.nkpts_par, self.dimC, self.dimC), dtype=np.complex128)
            if self.data.symmetry.write_gf
            else None
        )
        sgmL_k = (
            np.zeros((self.nkpts_par, self.dimC, self.dimC), dtype=np.complex128)
            if self.data.symmetry.write_lead_sgm
            else None
        )
        sgmR_k = (
            np.zeros((self.nkpts_par, self.dimC, self.dimC), dtype=np.complex128)
            if self.data.symmetry.write_lead_sgm
            else None
        )
        return gC_k, sgmL_k, sgmR_k

    def process_kpoint(self, ie_g: int, ik: int):
        hamiltonian_setup(
            ik=ik,
            ie_g=ie_g,
            egrid=self.egrid,
            shift_L=self.data.shift_L,
            shift_C=self.data.shift_C,
            shift_R=self.data.shift_R,
            shift_C_corr=getattr(self.data, "shift_corr", 0.0),
            blc_blocks=self.blc_blocks,
            ie_buff=1,
        )

        sigma_R, sigma_L, niter_R, niter_L = build_self_energies_from_blocks(
            blc_00R=self.blc_blocks["blc_00R"].at_k(ik),
            blc_01R=self.blc_blocks["blc_01R"].at_k(ik),
            blc_00L=self.blc_blocks["blc_00L"].at_k(ik),
            blc_01L=self.blc_blocks["blc_01L"].at_k(ik),
            blc_CR=self.blc_blocks["blc_CR"].at_k(ik),
            blc_LC=self.blc_blocks["blc_LC"].at_k(ik),
            leads_are_identical=self.data.advanced.leads_are_identical,
            delta=self.delta,
            niterx=self.data.iteration.niterx,
            transfer_thr=self.data.iteration.transfer_thr,
            fail_counter=None,
            fail_limit=self.data.iteration.nfailx,
            verbose=False,
        )

        gC = compute_conductor_green_function(
            blc_00C=self.blc_blocks["blc_00C"].at_k(ik),
            sigma_l=sigma_L,
            sigma_r=sigma_R if not self.data.advanced.surface else None,
            delta=self.delta,
            surface=self.data.advanced.surface,
        )

        niter_sum = niter_R + (
            niter_L if not self.data.advanced.leads_are_identical else 0
        )
        return gC, sigma_L, sigma_R, niter_sum

    def accumulate_dos(self, dos, dos_k, gC, ie_g, ik):
        diag_imag = np.imag(np.diagonal(gC))
        dos_k[ie_g, ik] = -self.wk_par[ik] * np.sum(diag_imag) / np.pi
        dos[ie_g] += dos_k[ie_g, ik]

    def accumulate_conductance(
        self, conduct, conduct_k, gC, sigma_L, sigma_R, ie_g, ik
    ):
        gamma_L = 1j * (sigma_L - sigma_L.conj().T)
        gamma_R = 1j * (sigma_R - sigma_R.conj().T)

        do_eigplot_now = (
            self.data.symmetry.do_eigenchannels
            and self.data.symmetry.do_eigplot
            and ie_g == self.data.symmetry.ie_eigplot
            and ik == self.data.symmetry.ik_eigplot
        )

        cond_aux, z_eigplot = evaluate_transmittance(
            gamma_L=gamma_L,
            gamma_R=gamma_R,
            G_ret=gC,
            formula=self.data.conduct_formula,
            do_eigenchannels=self.data.symmetry.do_eigenchannels,
            do_eigplot=do_eigplot_now,
            sgm_corr=None,
            eta=self.delta,
            S_overlap=None,
        )

        conduct[0, ie_g] += self.wk_par[ik] * np.sum(cond_aux)
        conduct_k[0, ik, ie_g] += self.wk_par[ik] * np.sum(cond_aux)

        if self.data.symmetry.do_eigenchannels:
            nchan = min(conduct.shape[0] - 1, cond_aux.shape[0])
            conduct[1 : 1 + nchan, ie_g] += self.wk_par[ik] * cond_aux[:nchan]
            conduct_k[1 : 1 + nchan, ik, ie_g] += self.wk_par[ik] * cond_aux[:nchan]

        if do_eigplot_now and z_eigplot is not None and self.rank == 0:
            write_eigenchannels(
                data=z_eigplot,
                ie=ie_g,
                ik=ik,
                vkpt=self.vkpt_par3D[:, ik],
                transport_direction=self.data.transport_direction,
                output_dir=Path("output/eigenchannels"),
                prefix="eigchn",
                overwrite=True,
                verbose=True,
            )

        conduct[0, ie_g] += self.wk_par[ik] * np.sum(cond_aux)
        conduct_k[0, ik, ie_g] += self.wk_par[ik] * np.sum(cond_aux)

        if self.data.symmetry.do_eigenchannels:
            nchan = min(conduct.shape[0] - 1, cond_aux.shape[0])
            conduct[1 : 1 + nchan, ie_g] += self.wk_par[ik] * cond_aux[:nchan]
            conduct_k[1 : 1 + nchan, ik, ie_g] += self.wk_par[ik] * cond_aux[:nchan]

    def finalize_energy(self, avg_iter, ie_g, gC_k, sgmL_k, sgmR_k):
        avg_iter /= 2 * self.nkpts_par
        if self.rank == 0:
            print(f"  T matrix converged after avg. # of iterations {avg_iter:10.3f}\n")
            global_timing.timing_upto_now(
                "do_conductor", label="Total time spent up to now"
            )

        if self.data.symmetry.write_gf:
            for ir in range(self.nrtot_par):
                self.gf_out[ie_g, ir] = compute_rham(
                    self.vr_par3D[:, ir], gC_k, self.vkpt_par3D.T, self.wk_par
                )
        if self.data.symmetry.write_lead_sgm:
            for ir in range(self.nrtot_par):
                self.rsgmL_out[ie_g, ir] = compute_rham(
                    self.vr_par3D[:, ir], sgmL_k, self.vkpt_par3D.T, self.wk_par
                )
                self.rsgmR_out[ie_g, ir] = compute_rham(
                    self.vr_par3D[:, ir], sgmR_k, self.vkpt_par3D.T, self.wk_par
                )

    def reduce_results(self, conduct, dos, conduct_k, dos_k):
        self.comm.Allreduce(MPI.IN_PLACE, conduct, op=MPI.SUM)
        self.comm.Allreduce(MPI.IN_PLACE, conduct_k, op=MPI.SUM)
        self.comm.Allreduce(MPI.IN_PLACE, dos, op=MPI.SUM)
        self.comm.Allreduce(MPI.IN_PLACE, dos_k, op=MPI.SUM)

        if self.data.symmetry.write_gf:
            self.comm.Allreduce(MPI.IN_PLACE, self.gf_out, op=MPI.SUM)
        if self.data.symmetry.write_lead_sgm:
            self.comm.Allreduce(MPI.IN_PLACE, self.rsgmL_out, op=MPI.SUM)
            self.comm.Allreduce(MPI.IN_PLACE, self.rsgmR_out, op=MPI.SUM)

    def write_operators(self):
        if self.rank != 0:
            return

        if self.data.symmetry.write_gf:
            write_operator_xml(
                "greenf.xml",
                self.gf_out,
                self.ivr_par3D,
                self.egrid,
                self.dimC,
                True,
                "retarded",
                "eV",
            )
        if self.data.symmetry.write_lead_sgm:
            write_operator_xml(
                "lead_L_sgm.xml",
                self.rsgmL_out,
                self.ivr_par3D,
                self.egrid,
                self.dimC,
                True,
                "retarded",
                "eV",
            )
            write_operator_xml(
                "lead_R_sgm.xml",
                self.rsgmR_out,
                self.ivr_par3D,
                self.egrid,
                self.dimC,
                True,
                "retarded",
                "eV",
            )
            write_kresolved_operator_xml(
                filename="lead_L_k.xml",
                dimwann=self.dimC,
                vkpt=self.vkpt_par3D,
            )

    def get_k_resolved_conductance(self, conduct_k: np.ndarray) -> np.ndarray:
        return conduct_k

    def get_k_resolved_dos(self, dos_k: np.ndarray) -> np.ndarray:
        return dos_k

    @headered_function("Writing data")
    def write_output(self):
        if self.rank != 0:
            return

        output_dir = Path(self.data.file_names.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        postfix = self.data.file_names.postfix

        write_data(self.egrid, self.conduct, "conductance", output_dir, postfix=postfix)
        write_data(self.egrid, self.dos, "doscond", output_dir, postfix=postfix)

        if self.data.symmetry.write_kdata:
            nkpts_par = self.data.get_runtime_data().nkpts_par
            prefix = os.path.basename(self.data.file_names.datafile_C)

            for ik in range(nkpts_par):
                ik_str = f"{ik + 1:04d}"
                filename_cond = f"{prefix}_cond-{ik_str}.dat"
                filename_dos = f"{prefix}_doscond-{ik_str}.dat"

                with (output_dir / filename_cond).open("w") as f:
                    for ie in range(self.egrid.shape[0]):
                        values = " ".join(
                            f"{self.conduct_k[ch, ik, ie]:15.9f}"
                            for ch in range(self.conduct_k.shape[0])
                        )
                        f.write(f"{self.egrid[ie]:15.9f} {values}\n")

                with (output_dir / filename_dos).open("w") as f:
                    for ie in range(self.egrid.shape[0]):
                        f.write(f"{self.egrid[ie]:15.9f} {self.dos_k[ie, ik]:15.9f}\n")
