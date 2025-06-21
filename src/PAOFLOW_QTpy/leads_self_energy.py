import numpy as np
from PAOFLOW_QTpy.green import compute_surface_green_function
from PAOFLOW_QTpy.transfer import compute_surface_transfer_matrices


def compute_lead_self_energy(
    h_eff: np.ndarray,
    s_eff: np.ndarray,
    h_coupling: np.ndarray,
    delta: float = 1e-5,
    direction: str = "right",
    niterx: int = 200,
    transfer_thr: float = 1e-12,
    fail_counter: dict = None,
    fail_limit: int = 10,
    verbose: bool = False,
) -> tuple[np.ndarray, int]:
    """
    Compute the self-energy matrix for a semi-infinite lead using surface Green's function.

    Parameters
    ----------
    `h_eff` : np.ndarray
        On-site Hamiltonian block H_00.
    `s_eff` : np.ndarray
        Overlap matrix S_00.
    `h_coupling` : np.ndarray
        Inter-cell hopping matrix H_01 (to the adjacent unit cell).
    `delta` : float
        Imaginary broadening parameter.
    `direction` : {'right', 'left'}
        Indicates which semi-infinite lead is being modeled.
    `niterx` : int
        Maximum number of iterations for Sancho-Rubio method.
    `transfer_thr` : float
        Convergence threshold for transfer matrices.
    `fail_counter` : dict
        Optional mutable dict to track convergence failures.
    `fail_limit` : int
        Maximum number of allowed convergence failures.
    `verbose` : bool
        Whether to log progress and warnings.

    Returns
    -------
    `sigma` : np.ndarray
        Self-energy matrix of the lead (same shape as h_eff).
    `niter` : int
        Number of iterations used in the Sancho-Rubio recursion.

    Notes
    -----
    This function constructs the lead self-energy Σ(E) for a semi-infinite system
    using the expression:

        Σ(E) = H_01† · G_surf(E) · H_01      (for right lead)
        Σ(E) = H_01 · G_surf(E) · H_01†      (for left lead)

    where:
    - G_surf(E) is the surface Green's function of the lead, defined as


        G_surf(E) = [E·S - H - H_01·T]⁻¹              (right surface)
                  = [E·S - H - H_01†·T†]⁻¹            (left surface)

    - T and T† are the lead transfer matrices computed using the Sancho-Rubio method.

    """

    tot, tott, niter = compute_surface_transfer_matrices(
        h_eff,
        s_eff,
        h_coupling,
        delta=delta,
        niterx=niterx,
        transfer_thr=transfer_thr,
        fail_counter=fail_counter,
        fail_limit=fail_limit,
        verbose=verbose,
    )

    igreen = 1 if direction == "right" else -1

    g_surf = compute_surface_green_function(
        h_eff, s_eff, h_coupling, tot, tott, igreen=igreen, delta=delta
    )

    if direction == "right":
        sigma = h_coupling.conj().T @ g_surf @ h_coupling
    else:
        sigma = h_coupling @ g_surf @ h_coupling.conj().T

    return sigma, niter


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
) -> tuple[np.ndarray, np.ndarray, int, int]:
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
    `niter_R` : int
        Iteration count for right lead.
    `niter_L` : int
        Iteration count for left lead.
    """
    gR, niter_R = compute_lead_self_energy(
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
        gL, niter_L = compute_lead_self_energy(
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
        gL, niter_L = compute_lead_self_energy(
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

    return sigma_R, sigma_L, niter_R, niter_L
