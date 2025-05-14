# FORTRAN to Python Mapping for `conductor.f90`

This document tracks the translation of Fortran routines into modular, Pythonic code.

---

## 🔁 Function Mapping Table

src/
| Fortran Function | Python Equivalent | Status | Notes |
| ------------------------ | ------------------------------------------- | ----------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `do_conductor` | `do_conductor` | In Progress | Will be split into setup, Green's function, transmittance, and output |
| `workspace_allocate` | Handled in `init_conductor_data()` | Removed | Replaced by numpy array allocations |
| `memusage`, `flush_unit` | Python `logging` module | Removed | Simplified and handled using standard logging |
| `transfer_mtrx` | `compute_surface_transfer_matrices` | Translated | Fully implemented using NumPy; follows Sancho-Rubio method |
| `green` | `compute_surface_green_function` | Translated | Handles `igreen = -1, 0, 1` for surface and bulk Green’s functions |
| - (new function) | `compute_lead_self_energy` | Added | Computes lead self-energies Σ using `transfer_mtrx` and `green` logic |
| - (new function) | `compute_conductor_green_function` | Added | Computes the retarded green's function |
| `gzero_maker` | `compute_non_interacting_gf` | Translated | Full implementation supporting lorentzian, none, and numerical smearing |
| `smearing_func` | `smearing_func` | Translated | Implements smearing types: lorentzian, gaussian, fermi-dirac, MP, MV |
| `smearing_init` | `initialize_smearing_grid` | Translated | FFT-based construction of xgrid and g_smear for numerical smearing |
| `input_parameters.f90` | Pass default values explicitly to functions | Removed | It would be cleaner to implement a `TransportConfig` dataclass that is passed to various functions. However, I think this would make the functions less readable and the docstrings not as explicit |
| `transmittance` | `evaluate_transmittance` | Translated | Supports Fisher-Lee and generalized formula. Uses `scipy.linalg.solve` instead of matrix inversion for numerical stability. Implements eigenchannel decomposition and eigenvector output for plotting when requested. |
baselib/util.f90

| Fortran Routine | Purpose                                  | Python Equivalent                | Status                |
| --------------- | ---------------------------------------- | -------------------------------- | --------------------- |
| `mat_mul`       | General matrix multiplication (with ops) | `@`, `.conj().T @`               | Covered               |
| `mat_sv`        | Solving linear systems `A x = b`         | `inv(...) @ ...` or `solve(...)` | Covered (via `solve`) |
| `mat_hdiag`     | Hermitian matrix diagonalization         | `numpy.linalg.eigh(...)`         | Covered               |

## 📌 Additional Notes

- Global variables from Fortran modules are being passed explicitly in Python.
- Obsolete or utility routines that are replaced with NumPy or Python standard libraries are removed.
- Function signatures in Python include type hints and docstrings following Google-style documentation.
