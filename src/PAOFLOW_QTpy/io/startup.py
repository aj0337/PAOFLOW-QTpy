import datetime

from mpi4py import MPI

from PAOFLOW_QTpy import __version__


def startup(main_name):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.datetime.now().strftime("%H:%M:%S")

    if rank == 0:
        print("=" * 70)
        print("              =                                            =")
        print("              =           Quantum Transport Code           =")
        print("              =     (https://aflowlib.org/src/paoflow/)    =")
        print("              =                                            =")
        print("=" * 70)
        print(f"Program <{main_name}>  v. {__version__}  starts ...")
        print(f"Date {current_date} at {current_time}")
        print(f"Number of MPI processes:    {size}")
