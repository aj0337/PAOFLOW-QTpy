from PAOFLOW_QTpy.io.input_from_file import input_from_file
from PAOFLOW_QTpy.io.input_parameters import read_namelist_input_conductor


def input_manager():
    input_from_file()
    read_namelist_input_conductor()
    # setup_control()
    # setup_io()
    # setup_egrid()
    # setup_smearing()
    # setup_hamiltonian()
    # setup_correlation()
    # setup_kpoints()
