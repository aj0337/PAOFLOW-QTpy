# PAOFLOW-QTpy

Python version of the Quantum transport calculations done by PAOFLOW.

# Installation

```bash
pip install -e .
```

# Usage

`mpirun -n <num_procs> python <script.py> <input_file.yaml> > logfile.out`

# Input flags

See [README_conductor.input](README_conductor.input) for details on input flags to compute conductance and DOS.

See [README_current.input](README_current.input) for details on input flags to compute current as a function of bias voltage.

# Examples

See the [examples](examples) folder for usage examples.

`analysis.ipynb` contains code to plot the results. It compares the results from the Python version with the Fortran version.

## Example01:

Demonstrates how to compute electron conductance and DOS using a bulk Al.

## Example02:

Demonstrates how to compute conductance through a two terminal device made of Alh leads and bridge.

## Example04:

Demonstrates how to compute phonon transport calculation on a 1-dimensional polymer (poly-acethylene).

### Job sequence

- run pw.x

  - pw.x < cnhn.scf.in

- run ph.x

  - ph.x < cnhn.ph.in

- run q2trans.x

  - q2trans.x < cnhn.q2trans.in

    This extracts the matrix of the interatomic force constants,
    and writes it in the format for conductor.x (ext.ham)

- compute the phonon transmittance

  - python main.py conductor.yaml > conductor.out

NOTE: the input parameters (ecutwfc etc.) or q-grid are somewhat under
converged for the sake of fast execution (CAVEAT: this test takes a
LONG time to run on a workstation). For the testing of the transport
part, `ext.ham` has been copied from the Fortran `tests` folder.

NOTE2: for accurate phonon calculations on these systems, ecutrho=20\*ecuwfc,
else instabilities appear in the long wavelength acoustic modes.

# Testing

```bash
pytest -v tests/test_*.py
```
