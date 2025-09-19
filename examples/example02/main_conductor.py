from mpi4py import MPI

from PAOFLOW_QTpy.transport.do_conductor import ConductorCalculator
from PAOFLOW_QTpy.parsers.parser_base import parse_args
from PAOFLOW_QTpy.workspace.prepare_data import (
    prepare_conductor,
    prepare_hamiltonian_blocks_and_leads,
    prepare_hamiltonian_system,
    prepare_kpoints,
    prepare_smearing,
    prepare_workspace,
)
from PAOFLOW_QTpy.smearing.smearing_base import smearing_func
from PAOFLOW_QTpy.utils.memusage import MemoryTracker
from PAOFLOW_QTpy.utils.timing import global_timing, timed_function

comm = MPI.COMM_WORLD

# TODO Bug fix: figure out why the first timing report is always zero (in both examples)


@timed_function("conductor")
def main():
    yaml_file = parse_args()
    data = prepare_conductor(yaml_file)
    memory_tracker = MemoryTracker()

    _ = prepare_smearing(smearing_func, memory_tracker)

    _ = prepare_kpoints(data, memory_tracker)

    ham_sys = prepare_hamiltonian_system(data, memory_tracker)

    prepare_hamiltonian_blocks_and_leads(data, ham_sys)

    _ = prepare_workspace(data, memory_tracker)

    calculator = ConductorCalculator(
        data=data,
        blc_blocks=ham_sys.blocks,
    )
    calculator.run()

    if comm.rank == 0:
        calculator.write_output()
        global_timing.report()
        memory_tracker.report(include_real_memory=True)


if __name__ == "__main__":
    main()
