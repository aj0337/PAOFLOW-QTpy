# FORTRAN to Python Mapping for `conductor.f90`

This document tracks the translation of Fortran routines into modular, Pythonic code.

---

## 🔁 Function Mapping Table

| Fortran Function         | Python Equivalent                  | Status      | Notes                                                                 |
| ------------------------ | ---------------------------------- | ----------- | --------------------------------------------------------------------- |
| `do_conductor`           | `do_conductor`                     | In Progress | Will be split into setup, Green's function, transmittance, and output |
| `transfer_mtrx`          | TBD                                | Pending     | Required to proceed beyond Green’s function lead construction         |
| `workspace_allocate`     | Handled in `init_conductor_data()` | Removed     | Replaced by numpy array allocations                                   |
| `memusage`, `flush_unit` | Python `logging` module            | Removed     | Simplified and handled using standard logging                         |

---

## 📌 Additional Notes

- Global variables from Fortran modules are being passed explicitly in Python.
- Obsolete or utility routines that are replaced with NumPy or Python standard libraries are removed.
- Function signatures in Python include type hints and docstrings following Google-style documentation.
