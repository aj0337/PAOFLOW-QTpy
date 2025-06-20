import os
import numpy as np
from mpi4py import MPI
from time import perf_counter

from PAOFLOW_QTpy.do_conductor import run_conductor
from PAOFLOW_QTpy.hamiltonian_init import initialize_hamiltonian_blocks
from PAOFLOW_QTpy.io.startup import startup
from PAOFLOW_QTpy.io.write_header import write_header
from PAOFLOW_QTpy.parsers.atmproj_tools import parse_atomic_proj
from PAOFLOW_QTpy.io.summary import print_summary
from PAOFLOW_QTpy.io.get_input_params import load_summary_data_from_yaml
from PAOFLOW_QTpy.kpoints import (
    compute_fourier_phase_table,
    initialize_meshsize,
    initialize_kpoints,
    initialize_r_vectors,
    kpoints_mask,
)
from PAOFLOW_QTpy.smearing_base import smearing_func
from PAOFLOW_QTpy.smearing_T import SmearingData
from PAOFLOW_QTpy.kpoints import KpointsData
from PAOFLOW_QTpy.utils.memusage import MemoryTracker
from PAOFLOW_QTpy.hamiltonian import HamiltonianSystem
from PAOFLOW_QTpy.workspace import Workspace

comm = MPI.COMM_WORLD


def main():
    yaml_file = "./conductor.yaml"
    data_dict = load_summary_data_from_yaml(yaml_file)
    datafile_C = data_dict["datafile_C"]
    datafile_L = data_dict.get("datafile_L", "")
    datafile_R = data_dict.get("datafile_R", "")
    datafile_L_sgm = data_dict.get("datafile_L_sgm", "")
    datafile_R_sgm = data_dict.get("datafile_R_sgm", "")

    prefix = os.path.basename(datafile_C)
    work_dir = "./al5.save"
    atmproj_sh = data_dict["atmproj_sh"]
    atmproj_thr = data_dict["atmproj_thr"]
    do_orthoovp = data_dict["do_orthoovp"]
    calculation_type = data_dict["calculation_type"]

    data_dict["nproc"] = comm.Get_size()

    startup("conductor.py")
    write_header("Conductor Initialization")

    hk_data = parse_atomic_proj(
        file_proj=datafile_C,
        work_dir=work_dir,
        prefix=prefix,
        atmproj_sh=atmproj_sh,
        atmproj_thr=atmproj_thr,
        do_orthoovp=do_orthoovp,
        write_intermediate=True,
    )

    data_dict["nk_par"], data_dict["nr_par"] = initialize_meshsize(
        nr_full=hk_data["nr"], transport_dir=data_dict["transport_dir"]
    )

    s_par = data_dict["s"][:2]
    nk_par3d = kpoints_mask(data_dict["nk_par"], 1, data_dict["transport_dir"])
    s_par3d = kpoints_mask(s_par, 0, data_dict["transport_dir"])
    nr_par3d = kpoints_mask(data_dict["nr_par"], 1, data_dict["transport_dir"])

    vkpt_par3D, wk_par = initialize_kpoints(
        data_dict["nk_par"],
        s_par=s_par,
        transport_dir=data_dict["transport_dir"],
        use_symm=data_dict["use_symm"],
    )
    ivr_par3D, wr_par = initialize_r_vectors(
        data_dict["nr_par"], data_dict["transport_dir"]
    )

    data_dict.update(
        {
            "nkpts_par": vkpt_par3D.shape[0],
            "nrtot_par": ivr_par3D.shape[0],
            "vkpt_par3D": vkpt_par3D.T,
            "wk_par": wk_par,
            "ivr_par3D": ivr_par3D.T,
            "wr_par": wr_par,
            "nk_par3d": nk_par3d,
            "s_par3d": s_par3d,
            "nr_par3d": nr_par3d,
        }
    )

    print_summary(data_dict)

    memory_tracker = MemoryTracker()

    smearing_data = SmearingData(smearing_func=smearing_func)
    smearing_data.initialize(
        smearing_type="lorentzian", delta=1e-5, delta_ratio=5e-3, xmax=25.0
    )
    memory_tracker.register_section(
        "smearing", smearing_data.memory_usage, is_allocated=True
    )

    kpoints_data = KpointsData()
    kpoints_data.vkpt_par3D = vkpt_par3D
    kpoints_data.wk_par = wk_par
    kpoints_data.ivr_par3D = ivr_par3D
    kpoints_data.wr_par = wr_par
    memory_tracker.register_section(
        "kpoints", kpoints_data.memory_usage, is_allocated=True
    )

    dimL = data_dict["dimL"]
    dimC = data_dict["dimC"]
    dimR = data_dict["dimR"]
    nkpts_par = data_dict["nkpts_par"]
    ham_sys = HamiltonianSystem(dimL, dimC, dimR, nkpts_par)
    memory_tracker.register_section(
        "hamiltonian data",
        lambda: ham_sys.memusage("ham"),
        is_allocated=ham_sys.allocated,
    )
    memory_tracker.register_section(
        "correlation data",
        lambda: ham_sys.memusage("corr"),
        is_allocated=ham_sys.allocated,
    )
    table_par = compute_fourier_phase_table(vkpts=vkpt_par3D, ivr_par=ivr_par3D)
    leads_are_identical = initialize_hamiltonian_blocks(
        ham_system=ham_sys,
        ivr_par3D=ivr_par3D.T,
        wr_par=wr_par,
        table_par=table_par,
        datafile_C=datafile_C,
        datafile_L=datafile_L,
        datafile_R=datafile_R,
        datafile_L_sgm=datafile_L_sgm,
        datafile_R_sgm=datafile_R_sgm,
        ispin=data_dict["ispin"],
        transport_dir=data_dict["transport_dir"],
        calculation_type=calculation_type,
    )
    data_dict["leads_are_identical"] = leads_are_identical

    workspace = Workspace()
    workspace.allocate(
        dimL=dimL,
        dimC=dimC,
        dimR=dimR,
        dimx_lead=max(dimL, dimR),
        nkpts_par=nkpts_par,
        nrtot_par=data_dict["nrtot_par"],
        write_lead_sgm=data_dict.get("write_lead_sgm", False),
        write_gf=data_dict.get("write_gf", False),
    )
    memory_tracker.register_section(
        "workspace", workspace.memusage, is_allocated=workspace.allocated
    )

    print(memory_tracker.report(include_real_memory=True))

    write_header("Frequency Loop")

    data_dict["_freqloop_start_time"] = perf_counter()
    conduct, dos = run_conductor(
        data_dict=data_dict,
        ne=data_dict["ne"],
        egrid=np.linspace(data_dict["emin"], data_dict["emax"], data_dict["ne"]),
        nkpts_par=nkpts_par,
        shifts={
            "shift_L": data_dict["shift_L"],
            "shift_C": data_dict["shift_C"],
            "shift_R": data_dict["shift_R"],
        },
        blc_blocks=ham_sys.blocks,
        wk_par=wk_par,
        vkpt_par3D=vkpt_par3D,
        transport_dir=data_dict["transport_dir"],
        conduct_formula=data_dict["conduct_formula"],
        do_eigenchannels=data_dict.get("do_eigenchannels", False),
        neigchnx=data_dict.get("neigchnx", 0),
        do_eigplot=data_dict.get("do_eigplot", False),
        ie_eigplot=data_dict.get("ie_eigplot", None),
        ik_eigplot=data_dict.get("ik_eigplot", None),
        leads_are_identical=leads_are_identical,
        surface=data_dict.get("surface", False),
        lhave_corr=data_dict.get("lhave_corr", False),
        ldynam_corr=data_dict.get("ldynam_corr", False),
        delta=data_dict["delta"],
        niterx=data_dict.get("niterx", 200),
        transfer_thr=data_dict.get("transfer_thr", 1e-12),
        nprint=data_dict.get("nprint", 20),
    )


if __name__ == "__main__":
    main()
