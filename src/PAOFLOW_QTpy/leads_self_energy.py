import numpy as np
from green import compute_lead_self_energy


def build_self_energies_from_blocks(
    blc_00R: np.ndarray,
    blc_01R: np.ndarray,
    blc_00L: np.ndarray,
    blc_01L: np.ndarray,
    blc_CR: np.ndarray,
    blc_LC: np.ndarray,
    leads_are_identical: bool,
    delta: float = 1e-5,
    niterx: int = 200,
    transfer_thr: float = 1e-12,
    fail_counter: dict | None = None,
    fail_limit: int = 10,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct lead self-energies Σ_R and Σ_L using Green's function recursion.

    Parameters
    ----------
    `blc_00R` : (nR, nR) complex ndarray
        On-site Hamiltonian of the right lead.
    `blc_01R` : (nR, nR) complex ndarray
        Hopping between right lead cells.
    `blc_00L` : (nL, nL) complex ndarray
        On-site Hamiltonian of the left lead.
    `blc_01L` : (nL, nL) complex ndarray
        Hopping between left lead cells.
    `blc_CR` : (nC, nR) complex ndarray
        Coupling block between conductor and right lead.
    `blc_LC` : (nC, nL) complex ndarray
        Coupling block between conductor and left lead.
    `leads_are_identical` : bool
        If True, reuse right lead calculation for left.
    `delta` : float
        Broadening parameter.
    `niterx` : int
        Max iterations for Sancho-Rubio method.
    `transfer_thr` : float
        Threshold for transfer matrix convergence.
    `fail_counter` : dict or None
        Shared dict for tracking failures across calls.
    `fail_limit` : int
        Maximum allowed failures.
    `verbose` : bool
        Enable logging information.

    Returns
    -------
    `sigma_R` : (nC, nC) complex ndarray
        Self-energy from the right lead.
    `sigma_L` : (nC, nC) complex ndarray
        Self-energy from the left lead.
    """
    gR = compute_lead_self_energy(
        blc_00R,
        np.eye(blc_00R.shape[0]),
        blc_01R,
        delta=delta,
        direction="right",
        niterx=niterx,
        transfer_thr=transfer_thr,
        fail_counter=fail_counter,
        fail_limit=fail_limit,
        verbose=verbose,
    )
    sigma_R = blc_CR @ gR @ blc_CR.conj().T

    if leads_are_identical:
        gL = compute_lead_self_energy(
            blc_00R,
            np.eye(blc_00R.shape[0]),
            blc_01R,
            delta=delta,
            direction="left",
            niterx=niterx,
            transfer_thr=transfer_thr,
            fail_counter=fail_counter,
            fail_limit=fail_limit,
            verbose=verbose,
        )
    else:
        gL = compute_lead_self_energy(
            blc_00L,
            np.eye(blc_00L.shape[0]),
            blc_01L,
            delta=delta,
            direction="left",
            niterx=niterx,
            transfer_thr=transfer_thr,
            fail_counter=fail_counter,
            fail_limit=fail_limit,
            verbose=verbose,
        )
    sigma_L = blc_LC.conj().T @ gL @ blc_LC

    return sigma_R, sigma_L
