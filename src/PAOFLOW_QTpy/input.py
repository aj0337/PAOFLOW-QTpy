from PAOFLOW_QTpy.input_parameters import ConductorData
from mpi4py import MPI


def input_manager(filename, comm):

    conductorData = ConductorData(filename, comm)
    input_dict = conductorData.dict()

    return input_dict


#comm = MPI.COMM_WORLD
#input_dict = input_manager('test.yaml', comm)
