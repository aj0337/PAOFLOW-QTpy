from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def log_rank0(message: str):
    if rank == 0:
        print(message)


def log_section_start(name: str):
    log_rank0(f"Begins {name}")


def log_section_end(name: str):
    log_rank0(f"Ends {name}")
