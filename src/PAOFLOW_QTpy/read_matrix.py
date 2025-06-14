from __future__ import annotations

import numpy as np

from PAOFLOW_QTpy.operators.operator_blc import OperatorBlock


def read_matrix_block(
    matrix: np.ndarray,
    operator: OperatorBlock,
    ivr_index: int | None = None,
    enforce_hermiticity: bool = True,
) -> None:
    """
    Assign a Hamiltonian matrix block to an OperatorBlock, with optional Hermiticity enforcement.

    This function sets the internal matrix `H` of the given `OperatorBlock` using the provided
    `matrix` array. It assumes that the input array is already shaped as (dim1, dim2, nkpts),
    and that the matrix corresponds to the correct R-vector component as required by the transport setup.

    Parameters
    ----------
    `matrix` : np.ndarray
        Complex-valued Hamiltonian block of shape (dim1, dim2, nkpts).

    `operator` : OperatorBlock
        Target operator block to store the matrix. Allocation is done internally
        based on the shape of `matrix`.

    `ivr_index` : int, optional
        If set, can be used to track or tag the R-vector index associated with this block.
        This is not used functionally here but retained for compatibility with higher-level logic.

    `enforce_hermiticity` : bool
        Whether to enforce Hermiticity of the matrix via symmetrization H ← ½(H + Hᴴ).

    Notes
    -----
    - This routine assumes that the matrix has already been parsed and shaped correctly.
    - The input matrix is copied into the operator, with optional Hermitian symmetrization.
    - No I/O or transformation is performed here — this is a memory-level assignment step.
    - The OperatorBlock is reallocated if not already allocated.
    """
    dim1, dim2, nkpts = matrix.shape

    if not operator.allocated:
        operator.allocate(dim1=dim1, dim2=dim2, nkpts=nkpts)

    if enforce_hermiticity:
        operator.H = 0.5 * (matrix + matrix.swapaxes(0, 1).conj())
    else:
        operator.H = matrix.copy()

    if ivr_index is not None:
        operator.ivr = np.array(ivr_index)
