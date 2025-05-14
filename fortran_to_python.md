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
| `smearing_init` | `initialize_smearing_grid` | Translated | FFT-based construction of xgrid and g*smear for numerical smearing |
| `input_parameters.f90` | Pass default values explicitly to functions | Removed | It would be cleaner to implement a `TransportConfig` dataclass that is passed to various functions. However, I think this would make the functions less readable and the docstrings not as explicit |
| `transmittance` | `evaluate_transmittance` | Translated | Supports Fisher-Lee and generalized formula. Uses `scipy.linalg.solve` instead of matrix inversion for numerical stability. Implements eigenchannel decomposition and eigenvector output for plotting when requested. |
| `operator_blc` (type and related procedures) | `OperatorBlock` (class) | Translated | Encapsulates Hamiltonian, overlap, and self-energy blocks for transport calculations. Handles allocation, cleanup, copying, memory usage, and diagnostics. |
| `operator_blc_init` | `__init__` | Translated | Initializes all metadata and allocates nothing. Ensures all internal structures are in a clean, unallocated state. |
| `operator_blc_allocate` | `allocate()` | Translated | Allocates internal arrays (Hamiltonian, overlap, auxiliary matrices, etc.) based on provided dimensions and flags. Validates dimensions and sets internal state. |
| `operator_blc_deallocate` | `deallocate()` | Translated | Frees allocated arrays and resets metadata. Ensures clean object reuse or destruction. |
| `operator_blc_copy` | `copy_from(other)` | Translated | Copies allocation and content from another `OperatorBlock`. Raises error if source is unallocated. |
| `operator_blc_update` | `update(...)` | Translated | Updates metadata like energy index, k-point index, or dynamic correlation flag. Intended for runtime state tracking. |
| `operator_blc_memusage` | `memory_usage(memtype)` | Translated | Computes memory usage (in MB) for Hamiltonian or correlation data stored in the object. |
| `operator_blc_write` | `summary(unit=sys.stdout)`| Translated | Prints a summary of the block's current state and metadata. Useful for debugging and diagnostics. |
| `T_hamiltonian_module` | `HamiltonianSystem` | Translated | Encapsulates dimensions, shifts, and all Hamiltonian-related blocks (`blc_00L`, `blc_01L`, etc.) as attributes. Replaces global state with an object-oriented structure for clarity and maintainability. |
| `hamiltonian_allocate` | `HamiltonianSystem.allocate` | Translated | Initializes each `OperatorBlock` (blc_00L, blc_01L, etc.) and computes `dimx`, `dimx_lead`. Uses internal validation to ensure proper dimension setup. |
| `hamiltonian_deallocate` | `HamiltonianSystem.deallocate` | Translated | Deallocates each block and resets the allocation flag. All internal memory-managed structures are cleared. |
| `hamiltonian_memusage` | `HamiltonianSystem.memusage` | Translated | Returns total memory usage (in MB) across all allocated blocks by summing the usage reported by each `OperatorBlock`. |
| `operator_read_aux` | `read_operator_aux` | Placeholder | Pending IOTK XML file structure example for implementation. |
| `operator_read_data` | `read_operator_data` | Placeholder | Same as above. Assumes XML layout with <OPR1>, <VR1> tags etc. |
| `operator_write_aux` | `write_operator_aux` | Placeholder | Pending structure of energy grid and IVR/VR blocks. |
| `operator_write_data` | `write_operator_data` | Placeholder | Will need to write 3D complex arrays in XML blocks per IOTK. |
| `fourier_par` | `fourier_transform_real_to_kspace` | Translated | Performs a 2D Fourier transform from real-space matrices `rh[i,j,R]` to reciprocal space matrices `kh[i,j,k]` using weights `wr[R]` and phase factors `table[R,k]`. Vectorized with NumPy for performance and clarity. |\**\*\*
| `hamiltonian_setup` | `hamiltonian_setup` | Translated | Initializes the auxiliary matrices `aux` as `(E - shift) _ S - H - Σ`for each allocated Hamiltonian block at a given energy and k-point. Supports optional correlation self-energies and correction shift for the central region. |
|`transfer_mtrx + green + mat_mul`|`build_self_energies_from_blocks` | Added | Encapsulates transfer matrix construction and Green's function inversion into a reusable, general-purpose self-energy builder. |
| `divide_et_impera` | `divide_work` | Translated | Divides a 1-indexed loop range evenly across MPI ranks. Matches the logic of the Fortran version including remainders. |
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
