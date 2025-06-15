from PAOFLOW_QTpy.io.startup import startup
from PAOFLOW_QTpy.io.write_header import write_header
from PAOFLOW_QTpy.parsers.atmproj_tools import parse_atomic_proj
from PAOFLOW_QTpy.io.summary import print_summary
from PAOFLOW_QTpy.io.get_input_params import load_summary_data_from_yaml
from PAOFLOW_QTpy.kpoints import initialize_kpoints, initialize_r_vectors

from PAOFLOW_QTpy.smearing_base import smearing_func
from PAOFLOW_QTpy.smearing_T import SmearingData
from PAOFLOW_QTpy.kpoints import KpointsData
from PAOFLOW_QTpy.utils.memusage import MemoryTracker
from PAOFLOW_QTpy.hamiltonian import HamiltonianSystem
from PAOFLOW_QTpy.workspace import Workspace

from mpi4py import MPI

comm = MPI.COMM_WORLD


def main():
    yaml_file = "./conductor.yaml"
    data_dict = load_summary_data_from_yaml(yaml_file)
    datafile_C = data_dict["datafile_C"]
    work_dir = "./al5.save"
    postfix = data_dict["postfix"]
    atmproj_sh = data_dict["atmproj_sh"]
    atmproj_thr = data_dict["atmproj_thr"]
    do_orthoovp = data_dict["do_orthoovp"]

    data_dict["nproc"] = comm.Get_size()

    startup("conductor.py")
    write_header("Conductor Initialization")

    parse_atomic_proj(
        file_proj=datafile_C,
        work_dir=work_dir,
        postfix=postfix,
        atmproj_sh=atmproj_sh,
        atmproj_thr=atmproj_thr,
        do_orthoovp=do_orthoovp,
        write_intermediate=True,
    )

    data_dict["nr_par"] = [1, 1]  # TODO: Compute dynamically if possible

    vkpt_par3D, wk_par = initialize_kpoints(
        data_dict["nk"],
        data_dict["s"],
        data_dict["transport_dir"],
        data_dict["use_symm"],
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
        }
    )
    data_dict["nkpts_par"] = 1  # TODO: Fix this hardcoded value later
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
    ham_sys.allocate()
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

    workspace = Workspace()
    workspace.allocate(
        dimL=data_dict["dimL"],
        dimC=data_dict["dimC"],
        dimR=data_dict["dimR"],
        dimx_lead=max(data_dict["dimL"], data_dict["dimR"]),
        nkpts_par=data_dict["nkpts_par"],
        nrtot_par=data_dict["nrtot_par"],
        write_lead_sgm=data_dict.get("write_lead_sgm", False),
        write_gf=data_dict.get("write_gf", False),
    )
    memory_tracker.register_section(
        "workspace", workspace.memusage, is_allocated=workspace.allocated
    )

    print(memory_tracker.report(include_real_memory=True))


if __name__ == "__main__":
    main()
