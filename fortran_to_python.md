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
| `smearing_init` | `initialize_smearing_grid` | Translated | FFT-based construction of xgrid and g\*smear for numerical smearing |
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
| `hamiltonian_allocate` | `HamiltonianSystem.allocate` | Translated | Initializes each `OperatorBlock` (blc*00L, blc_01L, etc.) and computes `dimx`, `dimx_lead`. Uses internal validation to ensure proper dimension setup. |
| `hamiltonian_deallocate` | `HamiltonianSystem.deallocate` | Translated | Deallocates each block and resets the allocation flag. All internal memory-managed structures are cleared. |
| `hamiltonian_memusage` | `HamiltonianSystem.memusage` | Translated | Returns total memory usage (in MB) across all allocated blocks by summing the usage reported by each `OperatorBlock`. |
| `operator_read_aux` | `read_operator_aux` | Placeholder | Pending IOTK XML file structure example for implementation. |
| `operator_read_data` | `read_operator_data` | Placeholder | Same as above. Assumes XML layout with <OPR1>, <VR1> tags etc. |
| `operator_write_aux` | `write_operator_aux` | Placeholder | Pending structure of energy grid and IVR/VR blocks. |
| `operator_write_data` | `write_operator_data` | Placeholder | Will need to write 3D complex arrays in XML blocks per IOTK. |
| `fourier_par` | `fourier_transform_real_to_kspace` | Translated | Performs a 2D Fourier transform from real-space matrices `rh[i,j,R]` to reciprocal space matrices `kh[i,j,k]` using weights `wr[R]` and phase factors `table[R,k]`. Vectorized with NumPy for performance and clarity. |\*\*\*\*
| `hamiltonian_setup` | `hamiltonian_setup` | Translated | Initializes the auxiliary matrices `aux` as `(E - shift) * S - H - Σ`for each allocated Hamiltonian block at a given energy and k-point. Supports optional correlation self-energies and correction shift for the central region. |
|`transfer_mtrx + green + mat_mul`|`build_self_energies_from_blocks`| Added | Encapsulates transfer matrix construction and Green's function inversion into a reusable, general-purpose self-energy builder. |
|`divide_et_impera`|`divide_work`| Translated | Divides a 1-indexed loop range evenly across MPI ranks. Matches the logic of the Fortran version including remainders. |
|`do_conductor`(DOS section) |`compute_dos`| Translated & Split | Computes both total and k-resolved DOS from the diagonal of the Green's function. Returned as`(dos_k, dos)`. |
| `T_kpoints_module` |`kpoints.py` | Translated | Contains logic for generating 2D k-point mesh orthogonal to transport direction, applying symmetry reduction, converting to 3D, and computing phase factors for Fourier transforms. Includes`initialize_kpoints`, `kpoints_rmask`, `kpoints_equivalent`, and `compute_fourier_phase_table`. |
| `wd_write_eigchn` |`write_eigenchannels` | Translated | Writes eigenchannel vectors`z_eigplot`to`.npz`format with metadata including`ie`, `ik`, `vkpt`, and `transport_dir`. Handles safe file output and logs location. |
| `atmproj_tools`| `parse_atomic_proj` | Translated | Top-level function that orchestrates parsing of`data-file.xml`and`atomic_proj.xml`, builds H(k), and optionally writes `.ham`and`.space`. |
| (new in Python) | `parse_data_file` | New | Extracts lattice vectors, reciprocal vectors, and atomic positions/types from`data-file.xml`. |
| (new in Python) | `parse_atomic_proj_xml` | New | Parses eigenvalues, k-points, projectors, and overlaps from`atomic_proj.xml`. |
| (new in Python) | `build_hamiltonian_from_proj`| New | Constructs the k-dependent Hamiltonian from eigenvalues and projectors. Optionally uses overlaps for non-orthogonal projectors. |
| (new in Python) |`write_internal_format_files`| New | Writes`.ham`XML output in internal PAOFLOW format for debugging. Encodes lattice, weights, and Hamiltonian blocks in real space. |
|`grids_get_rgrid`|`grids_get_rgrid`| Translated | Generates the real-space R-vector grid in crystal coordinates and assigns normalized weights. Ensures time-reversal symmetry by explicitly including`-R`points and halving corresponding weights when necessary. Returns total number of R-points, R-vectors, and weights. |
|`compute_rham`             |`compute_rham`    | Translated               | Computes real-space Hamiltonian block`H(R)`via inverse Fourier transform from reciprocal-space`H(k)`using`H(R) = ∑_k w_k · exp(-i k · R) · H(k)`. Ensures `k · R`is dimensionless by requiring both`k`and`R` in Cartesian coordinates. |

baselib/util.f90

| Fortran Routine | Purpose                                  | Python Equivalent                | Status                |
| --------------- | ---------------------------------------- | -------------------------------- | --------------------- |
| `mat_mul`       | General matrix multiplication (with ops) | `@`, `.conj().T @`               | Covered               |
| `mat_sv`        | Solving linear systems `A x = b`         | `inv(...) @ ...` or `solve(...)` | Covered (via `solve`) |
| `mat_hdiag`     | Hermitian matrix diagonalization         | `numpy.linalg.eigh(...)`         | Covered               |

## Parsing

### QE XML Parsing Correspondence: Fortran vs Python

This section maps how QE XML tags are parsed in the Fortran module `qexml.f90` versus how they are handled in the Python-based parser (`parse_data_file` and `parse_atomic_proj_xml`).

---

### 🧭 Overview: QE XML Parsing Layers

| Data Type / Section        | XML Tag                      | Fortran (qexml.f90)                          | Python Function               | Notes                                                               |
| -------------------------- | ---------------------------- | -------------------------------------------- | ----------------------------- | ------------------------------------------------------------------- |
| Lattice parameter          | `LATTICE_PARAMETER`          | `qexml_read_cell`                            | `parse_data_file()`           | Value stored in `alat`                                              |
| Direct lattice vectors     | `DIRECT_LATTICE_VECTORS`     | `qexml_read_cell`                            | `parse_data_file()`           | Tags: `a1`, `a2`, `a3`                                              |
| Reciprocal lattice vectors | `RECIPROCAL_LATTICE_VECTORS` | `qexml_read_cell`                            | `parse_data_file()`           | Tags: `b1`, `b2`, `b3`                                              |
| Atomic positions           | `IONS`                       | `qexml_read_ions`                            | `parse_data_file()`           | Each atom has tag `ATOM`, child `TAU` and attributes like `SPECIES` |
| K-points                   | `K-POINTS`                   | `qexml_read_bz`                              | `parse_atomic_proj_xml()`     | Stored as `kpts`                                                    |
| K-point weights            | `WEIGHT_OF_K-POINTS`         | `qexml_read_bz`                              | `parse_atomic_proj_xml()`     | Stored as `wk`                                                      |
| Eigenvalues                | `EIGENVALUES` → `<EIG>`      | `qexml_read_bands`                           | `parse_atomic_proj_xml()`     | Stored as `eigvals`                                                 |
| Fermi energy               | `FERMI_ENERGY`               | `qexml_read_bands_info`                      | `parse_atomic_proj_xml()`     | Stored as `efermi`                                                  |
| Projections                | `PROJECTIONS`                | (Not shown in detail, implied in `read_wfc`) | `parse_atomic_proj_xml()`     | Used to build `proj[:, ib, ik, isp]`                                |
| Overlaps (optional)        | `OVERLAPS`                   | (Not shown in detail, implied in `read_wfc`) | `parse_atomic_proj_xml()`     | Optional, stored as `overlap`                                       |
| Units                      | e.g. `UNITS_FOR_ENERGIES`    | `iotk_scan_attr(...)`                        | Python uses `attrib['UNITS']` | Handled via attribute access in both                                |

---

### 🔄 One-to-One Tag/Field Mapping

#### From `parse_data_file` (Python)

| XML Element                            | Python Variable  | Fortran Subroutine | Fortran Variable |
| -------------------------------------- | ---------------- | ------------------ | ---------------- |
| `CELL/LATTICE_PARAMETER` (attr `alat`) | `alat`           | `qexml_read_cell`  | `alat`           |
| `CELL/A1`, `A2`, `A3`                  | `a1`, `a2`, `a3` | `qexml_read_cell`  | `a1`, `a2`, `a3` |
| `CELL/B1`, `B2`, `B3`                  | `b1`, `b2`, `b3` | `qexml_read_cell`  | `b1`, `b2`, `b3` |
| `IONS/ATOM`                            | `tau[:, i]`      | `qexml_read_ions`  | `tau(:, i)`      |
| `IONS/ATOM/@SPECIES`                   | `atm_symb[i]`    | `qexml_read_ions`  | `atm(i)`         |
| `IONS/ATOM/@ITYP`                      | `ityp[i]`        | `qexml_read_ions`  | `ityp(i)`        |

#### From `parse_atomic_proj_xml` (Python)

| XML Element                        | Python Variable          | Fortran Subroutine           | Fortran Variable         |
| ---------------------------------- | ------------------------ | ---------------------------- | ------------------------ |
| `HEADER/NUMBER_OF_BANDS`           | `nbnd`                   | `qexml_read_bands_info`      | `nbnd`                   |
| `HEADER/NUMBER_OF_K-POINTS`        | `nkpts`                  | `qexml_read_bands_info`      | `num_k_points`           |
| `HEADER/NUMBER_OF_SPIN_COMPONENTS` | `nspin`                  | `qexml_read_bands_info`      | `nspin`                  |
| `HEADER/NUMBER_OF_ATOMIC_WFC`      | `natomwfc`               | `qexml_read_bands_info`      | `natomwfc`               |
| `HEADER/FERMI_ENERGY`              | `efermi`                 | `qexml_read_bands_info`      | `ef`                     |
| `HEADER/UNITS_FOR_ENERGY/@UNITS`   | `energy_units`           | `qexml_read_bands_info`      | `energy_units`           |
| `K-POINTS`                         | `kpts`                   | `qexml_read_bz`              | `xk`                     |
| `WEIGHT_OF_K-POINTS`               | `wk`                     | `qexml_read_bz`              | `wk`                     |
| `EIGENVALUES/EIG`                  | `eigvals`                | `qexml_read_bands`           | `eig`                    |
| `PROJECTIONS/SPIN.../ATMWFC...`    | `proj[ia, ib, ik, isp]`  | implicit in `qexml_read_wfc` | `evc` projection vectors |
| `OVERLAPS/OVERLAP...`              | `overlap[:, :, ik, isp]` | optional in Fortran          | `overlap` matrices       |

---

### 🧾 Namespace Notes

- **Python**: XML namespace resolution is done using:
  ```python
  ns = {"qes": "http://www.quantum-espresso.org/ns/qes"}
  ```

## 📌 Additional Notes

- Global variables from Fortran modules are being passed explicitly in Python.
- Obsolete or utility routines that are replaced with NumPy or Python standard libraries are removed.
- Function signatures in Python include type hints and docstrings following Google-style documentation.
