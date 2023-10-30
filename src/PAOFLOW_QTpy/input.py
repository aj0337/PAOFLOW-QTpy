# from PAOFLOW_QTpy.input_parameters import ConductorData
# from mpi4py import MPI

# def input_manager(filename, comm):

#     conductorData = ConductorData(filename, comm)
#     input_dict = conductorData.dict()

#     return input_dict

#comm = MPI.COMM_WORLD
#input_dict = input_manager('test.yaml', comm)


def input_from_file(stdin):
    # Code for reading input from file
    pass


def read_namelist_input_conductor(stdin):
    # Code for reading and checking namelists
    pass


def setup_control():
    # Code for setting up control parameters
    pass


def setup_io():
    # Code for setting up input/output parameters
    pass


def setup_egrid():
    # Code for setting up energy grid
    pass


def setup_smearing():
    # Code for setting up smearing parameters
    pass


def setup_hamiltonian():
    # Code for setting up Hamiltonian parameters
    pass


def setup_kpoints():
    # Code for setting up k-points parameters
    pass


def setup_correlation():
    # Code for setting up correlation parameters
    pass


def input_manager():
    input_from_file(stdin)
    read_namelist_input_conductor(stdin)
    setup_control()
    setup_io()
    setup_egrid()
    setup_smearing()
    setup_hamiltonian()
    setup_correlation()
    setup_kpoints()


input_manager()
