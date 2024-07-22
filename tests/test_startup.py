import sys
from io import StringIO
from mpi4py import MPI

from PAOFLOW_QTpy.startup import startup


def test_startup():
    buffer = StringIO()
    sys.stdout = buffer
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    startup("test")
    output = buffer.getvalue()
    assert "test" in output
    assert f"Number of MPI processes:    {size}" in output
