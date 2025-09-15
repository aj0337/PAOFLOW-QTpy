import os
import sys
from pathlib import Path
import numpy as np
from mpi4py import MPI
from time import perf_counter

from PAOFLOW_QTpy.do_conductor import ConductorCalculator
from PAOFLOW_QTpy.hamiltonian_init import (
    check_leads_are_identical,
    initialize_hamiltonian_blocks,
)
from PAOFLOW_QTpy.io.startup import startup
from PAOFLOW_QTpy.io.write_header import write_header
from PAOFLOW_QTpy.parsers.atmproj_tools import parse_atomic_proj
from PAOFLOW_QTpy.io.summary import print_summary
from PAOFLOW_QTpy.io.get_input_params import load_conductor_data_from_yaml
from PAOFLOW_QTpy.io.input_parameters import RuntimeData
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
from PAOFLOW_QTpy.utils.timing import global_timing, timed_function

comm = MPI.COMM_WORLD


def parse_args():
    if len(sys.argv) != 2:
        if comm.rank == 0:
            print("Usage: python main.py <yaml_file>")
        sys.exit(1)
    return sys.argv[1]


@timed_function("conductor")
def main():
    yaml_file = parse_args()
    data = load_conductor_data_from_yaml(yaml_file)

    prefix = os.path.basename(data.file_names.datafile_C)
    work_dir = data.file_names.work_dir
    atmproj_sh = data.atomic_proj.atmproj_sh
    atmproj_thr = data.atomic_proj.atmproj_thr
    do_orthoovp = data.atomic_proj.do_orthoovp
    calculation_type = data.calculation_type
    nproc = comm.Get_size()

    startup("conductor.py")
    write_header("Conductor Initialization")

    hk_data = parse_atomic_proj(
        file_proj=data.file_names.datafile_C,
        work_dir=work_dir,
        prefix=prefix,
        atmproj_sh=atmproj_sh,
        atmproj_thr=atmproj_thr,
        do_orthoovp=do_orthoovp,
        write_intermediate=True,
    )

    nk_par, nr_par = initialize_meshsize(
        nr_full=hk_data["nr"], transport_dir=data.transport_dir
    )

    s_par = data.kpoint_grid.s[:2]
    nk_par3d = kpoints_mask(nk_par, 1, data.transport_dir)
    s_par3d = kpoints_mask(s_par, 0, data.transport_dir)
    nr_par3d = kpoints_mask(nr_par, 1, data.transport_dir)

    vkpt_par3D, wk_par = initialize_kpoints(
        nk_par,
        s_par=s_par,
        transport_dir=data.transport_dir,
        use_sym=data.symmetry.use_sym,
    )
    ivr_par3D, wr_par = initialize_r_vectors(nr_par, data.transport_dir)

    data.set_runtime_data(
        runtime=RuntimeData(
            nproc=nproc,
            prefix=prefix,
            work_dir=work_dir,
            nk_par=nk_par,
            s_par=s_par,
            nk_par3d=nk_par3d,
            s_par3d=s_par3d,
            nr_par3d=nr_par3d,
            vkpt_par3D=vkpt_par3D.T,
            wk_par=wk_par,
            ivr_par3D=ivr_par3D.T,
            wr_par=wr_par,
            nkpts_par=vkpt_par3D.shape[0],
            nrtot_par=ivr_par3D.shape[0],
        )
    )
    print_summary(data)

    memory_tracker = MemoryTracker()

    smearing_data = SmearingData(smearing_func=smearing_func)
    smearing_data.initialize()
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

    dimL, dimC, dimR = data.dimL, data.dimC, data.dimR
    nkpts_par = data.get_runtime_data().nkpts_par

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

    initialize_hamiltonian_blocks(
        ham_system=ham_sys,
        ivr_par3D=ivr_par3D.T,
        wr_par=wr_par,
        table_par=table_par,
        datafile_C=data.file_names.datafile_C,
        datafile_L=data.file_names.datafile_L,
        datafile_R=data.file_names.datafile_R,
        ispin=data.advanced.ispin,
        transport_dir=data.transport_dir,
        calculation_type=calculation_type,
    )

    data.advanced.leads_are_identical = check_leads_are_identical(
        ham_system=ham_sys,
        datafile_L=data.file_names.datafile_L,
        datafile_R=data.file_names.datafile_R,
        datafile_L_sgm=data.file_names.datafile_L_sgm,
        datafile_R_sgm=data.file_names.datafile_R_sgm,
    )

    workspace = Workspace()
    workspace.allocate(
        dimL=dimL,
        dimC=dimC,
        dimR=dimR,
        dimx_lead=max(dimL, dimR),
        nkpts_par=data.get_runtime_data().nkpts_par,
        nrtot_par=data.get_runtime_data().nrtot_par,
        write_lead_sgm=data.symmetry.write_lead_sgm,
        write_gf=data.symmetry.write_gf,
    )
    memory_tracker.register_section(
        "workspace", workspace.memusage, is_allocated=workspace.allocated
    )

    print(memory_tracker.report(include_real_memory=True))

    write_header("Frequency Loop")

    data._freqloop_start_time = perf_counter()

    egrid = np.linspace(data.energy.emin, data.energy.emax, data.energy.ne)
    calculator = ConductorCalculator(
        data=data,
        blc_blocks=ham_sys.blocks,
        egrid=egrid,
        wk_par=wk_par,
        vkpt_par3D=vkpt_par3D,
    )
    calculator.run()

    write_header("Writing data")

    if comm.rank == 0:
        output_dir = Path("output")
        calculator.write_output(output_dir, postfix=data.file_names.postfix)

    if comm.rank == 0:
        global_timing.report()


if __name__ == "__main__":
    main()
