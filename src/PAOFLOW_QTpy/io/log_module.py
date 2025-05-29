import logging
from datetime import datetime
import psutil

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Initialize the logger
logger = logging.getLogger("my_logger")
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler("logfile.log")
formatter = logging.Formatter("%(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Global variables
debug_mode = True  # Assuming this is set somewhere in your application
alloc = True  # Assuming this is set somewhere in your application
stack = []
stack_maxdim = 100  # Define the maximum stack size
stack_index = 0


def log_rank0(message: str):
    if rank == 0:
        print(message)


def log_push(name):
    global stack, stack_index

    # Check debug mode and allocation status
    if not debug_mode or not alloc:
        return

    # Update stack index and check bounds
    stack_index += 1
    if stack_index > stack_maxdim:
        return

    # Update stack with the new name
    if stack_index >= len(stack):
        stack.append("")
    stack[stack_index] = name.strip()

    # Get current date and time
    now = datetime.now()
    cdate = now.strftime("%Y-%m-%d")
    ctime = now.strftime("%H:%M:%S")

    # Get memory usage
    memory = psutil.Process().memory_info().rss // 1024  # Memory in KB

    # Format the log entry
    log_entry = (
        f"{cdate} {ctime} | {memory} KB | "
        + "..| " * (stack_index - 1)
        + stack[stack_index]
    )

    # Write to the log file
    logger.info(log_entry)
