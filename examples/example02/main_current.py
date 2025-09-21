from mpi4py import MPI

from PAOFLOW_QTpy.parsers.parser_base import parse_args
from PAOFLOW_QTpy.transport.do_current import CurrentCalculator
from PAOFLOW_QTpy.workspace.prepare_data import prepare_current
from PAOFLOW_QTpy.utils.memusage import MemoryTracker
from PAOFLOW_QTpy.utils.timing import global_timing, timed_function

comm = MPI.COMM_WORLD


@timed_function("current")
def main():
    yaml_file = parse_args()
    data = prepare_current(yaml_file)
    memory_tracker = MemoryTracker()

    if data is None:
        if comm.rank == 0:
            print("No current.yaml found. Skipping current calculation.")
        return

    calculator = CurrentCalculator(data)
    calculator.run()

    if comm.rank == 0:
        calculator.write_output()
        global_timing.report()
        memory_tracker.report(include_real_memory=True)


if __name__ == "__main__":
    main()
