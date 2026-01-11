from mpi4py import MPI

from PAOFLOW_QTpy.parsers.parser_base import parse_args
from PAOFLOW_QTpy.transport.do_conductor import ConductorRunner
from PAOFLOW_QTpy.utils.timing import global_timing, timed_function

comm = MPI.COMM_WORLD


@timed_function("conductor")
def main():
    yaml_file = parse_args()
    runner = ConductorRunner.from_yaml(yaml_file)
    runner.calculator.run()

    if comm.rank == 0:
        runner.calculator.write_output()
        global_timing.report()
        runner.memory_tracker.report(include_real_memory=True)


if __name__ == "__main__":
    main()
