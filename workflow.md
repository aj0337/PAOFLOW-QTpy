# Workflow: From Hamiltonian Creation to Transport Calculations in PAOFLOW-QTpy

## 1. **Hamiltonian Creation by Quantum ESPRESSO (QE)**

- **Physics**: QE generates the Hamiltonian in the form of Wannier-like projections or atomic wavefunctions. This Hamiltonian contains:
  - **K-points**: Sampling points in reciprocal space.
  - **Band structure**: Energy eigenvalues for each k-point.
  - **Atomic wavefunctions**: Projections of wavefunctions onto atomic orbitals.
- **Output**: Files such as `atomic_proj.xml` store this data.

**Relevant File**:

- `atomic_proj.xml` (example: `examples/example01/al5.save/atomic_proj.xml`)

---

## 2. **Post-Processing in PAOFLOW-QTpy**

### **a. K-Point Selection for Transport Directions**

- **Physics**: K-points are filtered to be orthogonal with the transport direction, ensuring proper sampling of electronic states contributing to transport.
- **Implementation**:
  - The file `src/grid/kpoints.py` handles k-point grids and filtering.
  - Functions likely manipulate or generate k-points orthogonal with the transport direction.

**Relevant File**:

- [`src/grid/kpoints.py`](src/grid/kpoints.py)

---

### **b. Sancho-Rubio Algorithm for Surface Green's Functions**

- **Physics**: The Sancho-Rubio iterative method computes the surface Green's functions of semi-infinite leads. These Green's functions are essential for connecting the device region to the leads.
- **Implementation**:
  - The file `src/transport/leads_self_energy.py` implements the Sancho-Rubio algorithm.
  - Iterative routines solve for the Green's function of a semi-infinite system.

**Relevant File**:

- [`src/transport/leads_self_energy.py`](src/transport/leads_self_energy.py)

---

### **c. Green's Function Calculation**

- **Physics**: The Green's function describes electron propagation in the system. It is used to compute the transmission function and other transport properties.
- **Implementation**:
  - The file `src/transport/green.py` calculates:
    - The retarded Green's function for the device region.
    - Coupling between the device and the leads.
  - Look for matrix inversion routines or Dyson's equation implementations.

**Relevant File**:

- [`src/transport/green.py`](src/transport/green.py)

---

### **d. Transmission and Conductance**

- **Physics**: The transmission function \( T(E) \) is computed using the Green's function and coupling matrices. Conductance is derived using the Landauer formula:
  \[
  G = \frac{2e^2}{h} \int T(E) \left(-\frac{\partial f}{\partial E}\right) dE
  \]
- **Implementation**:
  - The file `src/transport/transmittance.py` computes the transmission function \( T(E) \).
  - Look for routines that calculate the trace of matrices involving the Green's function and coupling matrices.

**Relevant File**:

- [`src/transport/transmittance.py`](src/transport/transmittance.py)

---

## 3. **Workflow Summary**

- **Input Parsing**:
  - The file `src/io/get_input_params.py` reads input parameters, including the transport direction and k-point grid.
- **Hamiltonian Setup**:
  - The file `src/hamiltonian/hamiltonian_setup.py` sets up the Hamiltonian matrix from QE outputs like `atomic_proj.xml`.
- **K-Point Filtering**:
  - The file `src/grid/kpoints.py` selects k-points parallel to the transport direction.
- **Green's Function and Self-Energy**:
  - The files `src/transport/leads_self_energy.py` and `src/transport/green.py` compute the Green's functions and self-energies.
- **Transmission and Conductance**:
  - The file `src/transport/transmittance.py` computes the transmission function and conductance using the Landauer formula.

---

## 4. **Relevant Physics and Code Sections**

| **Physics Step**             | **Code File**                          | **Description**                                                                |
| ---------------------------- | -------------------------------------- | ------------------------------------------------------------------------------ |
| K-point selection            | `src/grid/kpoints.py`                  | Handles k-point grids and filtering for transport directions.                  |
| Hamiltonian setup            | `src/hamiltonian/hamiltonian_setup.py` | Constructs the Hamiltonian matrix from QE outputs.                             |
| Sancho-Rubio algorithm       | `src/transport/leads_self_energy.py`   | Computes surface Green's functions for semi-infinite leads.                    |
| Green's function calculation | `src/transport/green.py`               | Calculates the Green's function for the device region.                         |
| Transmission and conductance | `src/transport/transmittance.py`       | Computes the transmission function and conductance using the Landauer formula. |

---

## 5. **References**

- **Hamiltonian Creation**: `atomic_proj.xml` (QE output)
- **PAOFLOW-QTpy Modules**:
  - `src/grid/kpoints.py`
  - `src/transport/leads_self_energy.py`
  - `src/transport/green.py`
  - `src/transport/transmittance.py`
  - `src/hamiltonian/hamiltonian_setup.py`
