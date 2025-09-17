import numpy as np
from typing import Literal


def initialize_energy_grid(
    emin: float,
    emax: float,
    ne: int,
    mode: Literal["electrons", "phonons"] = "electrons",
) -> np.ndarray:
    """
    Initialize a linear energy grid for transport calculations.

    Parameters
    ----------
    `emin` : float
        Minimum energy (eV or other consistent unit).
    `emax` : float
        Maximum energy (eV or other consistent unit).
    `ne` : int
        Number of energy points.
    `mode` : {'electrons', 'phonons'}
        Whether this is for electronic or phononic transport.

    Returns
    -------
    `egrid` : ndarray of shape (ne,)
        The energy grid values.

    Notes
    -----
    For 'phonons', the first energy is replaced with a small positive value
    to avoid numerical singularities at E = 0:
        egrid[0] = egrid[1] / 100.0
    """
    if ne <= 1:
        raise ValueError("Energy grid must have at least 2 points.")

    egrid = np.linspace(emin, emax, ne)

    if mode == "phonons":
        if egrid[0] == 0.0:
            egrid[0] = egrid[1] / 100.0

    return egrid
