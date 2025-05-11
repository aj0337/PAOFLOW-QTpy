import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def initialize_conductor_data(
    dim_c: int,
    dim_l: int,
    dim_r: int,
    nkpts: int,
    ne: int,
    neigchnx: int,
    do_eigenchannels: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Allocate and initialize data arrays for the conductor calculation.

    Parameters
    ----------
    `dim_c` : `int`
        Size of conductor region.
    `dim_l` : `int`
        Size of left lead.
    `dim_r` : `int`
        Size of right lead.
    `nkpts` : `int`
        Number of parallel k-points.
    `ne` : `int`
        Number of energy points.
    `neigchnx` : `int`
        Max number of eigenchannels to resolve.
    `do_eigenchannels` : `bool`
        Whether to compute eigenchannels.

    Returns
    -------
    `Tuple`:
        conduct_k, conduct, dos_k, dos, cond_aux arrays.
    """
    neigchn = min(dim_c, dim_r, dim_l, neigchnx) if do_eigenchannels else 0
    try:
        conduct_k = np.zeros((1 + neigchn, nkpts, ne), dtype=np.float64)
        conduct = np.zeros((1 + neigchn, ne), dtype=np.float64)
        dos_k = np.zeros((ne, nkpts), dtype=np.float64)
        dos = np.zeros(ne, dtype=np.float64)
        cond_aux = np.zeros(dim_c, dtype=np.float64)
    except MemoryError:
        logger.error("Memory allocation failed for conductance or DOS arrays.")
        raise

    return conduct_k, conduct, dos_k, dos, cond_aux
