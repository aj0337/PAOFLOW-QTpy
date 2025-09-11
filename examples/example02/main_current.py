import sys
from pathlib import Path
import numpy as np
from mpi4py import MPI

from PAOFLOW_QTpy.do_current import (
    read_transmittance,
    build_bias_grid,
    compute_current_vs_bias,
)
from PAOFLOW_QTpy.io.get_input_params import load_current_data_from_yaml

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def main():
    yaml_path = Path("current.yaml")
    inputs = load_current_data_from_yaml(yaml_path)

    if inputs is None:
        if rank == 0:
            print("No current.yaml found. Skipping current calculation.")
        return

    if rank == 0:
        print(f"Reading transmittance data from {inputs['filein']}")

    egrid, transm = read_transmittance(inputs["filein"])
    vgrid = build_bias_grid(inputs["Vmin"], inputs["Vmax"], inputs["nV"])

    currents = compute_current_vs_bias(
        egrid,
        transm,
        vgrid,
        inputs["mu_L"],
        inputs["mu_R"],
        inputs["sigma"],
    )

    if rank == 0:
        outpath = Path(inputs["fileout"])
        outpath.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(outpath, np.column_stack([vgrid, currents]))
        print(f"Saved current vs bias to {outpath}")


if __name__ == "__main__":
    sys.exit(main())
