def main():
    """"""
    import numpy as np
    from mpi4py import MPI

    from PAOFLOW_QTpy.input import input_manager
    from PAOFLOW_QTpy.smearing_T import define_input_pole
    comm = MPI.COMM_WORLD
    input_dict = input_manager('../../tests/test.yaml', comm)
    xmax = input_dict['xmax']
    delta_ratio = input_dict['delta_ratio']
    auxp_in = define_input_pole(xmax, delta_ratio)
    np.savetxt(
        '/home/anooja/Dropbox/anooja/UNT/PAOFLOW-QTpy/tests/auxp_in.dat',
        auxp_in)


if __name__ == "__main__":
    main()
