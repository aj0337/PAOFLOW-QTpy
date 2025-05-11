from typing import Literal, Optional

import numpy as np
from transfer import compute_surface_transfer_matrices


def compute_surface_green_function(
    h_eff: np.ndarray,
    s_eff: np.ndarray,
    t_coupling: np.ndarray,
    tot: np.ndarray,
    tott: np.ndarray,
    igreen: Literal[-1, 0, 1],
    delta: float = 1e-5,
) -> np.ndarray:
    """
    Construct the surface or bulk Green's function using the transfer matrices.

    Parameters
    ----------
    `h_eff` : np.ndarray
        Hamiltonian block `H_00` of shape (dim, dim).
    `s_eff` : np.ndarray
        Overlap matrix `S_00` of shape (dim, dim).
    `t_coupling` : np.ndarray
        Coupling matrix `H_01` of shape (dim, dim).
    `tot` : np.ndarray
        Right-going transfer matrix `T` of shape (dim, dim).
    `tott` : np.ndarray
        Left-going transfer matrix `T†` of shape (dim, dim).
    `igreen` : {-1, 0, 1}
        Green’s function type:
        -1: left surface,
         0: bulk,
         1: right surface.
    `delta` : float
        Small imaginary shift to stabilize inversion.

    Returns
    -------
    `green` : np.ndarray
        The computed Green’s function matrix `G(E)` of shape (dim, dim).

    Notes
    -----
    Implements the iterative method of Lopez Sancho et al. (J. Phys. F: Met. Phys., 14, 1205, 1984)
    to compute the transfer matrices `T` and `T†` for semi-infinite leads.

    The Green’s function is stabilized using a small imaginary part `delta`:
    `G = [E⋅S - H + i⋅delta⋅S - Σ]⁻¹`
    `T` and `T†` are constructed iteratively to capture surface coupling.

    """
    z_shift = 1j * delta * s_eff
    A = h_eff + z_shift

    if igreen == 1:
        A -= t_coupling @ tot
    elif igreen == -1:
        A -= t_coupling.conj().T @ tott
    elif igreen == 0:
        A -= t_coupling @ tot
        A -= t_coupling.conj().T @ tott
    else:
        raise ValueError(f"Invalid value for `igreen`: {igreen}. Must be -1, 0, or 1.")

    try:
        g = np.linalg.inv(A)
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            "Green's function inversion failed due to singular matrix."
        ) from e

    return g


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
) -> np.ndarray:
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

    tot, tott, _ = compute_surface_transfer_matrices(
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

    return sigma


def compute_conductor_green_function(
    energy: float,
    h_c: np.ndarray,
    s_c: np.ndarray,
    sigma_l: np.ndarray,
    sigma_r: np.ndarray,
    surface: bool = False,
    smearing_type: str = "lorentzian",
    delta: float = 1e-5,
    delta_ratio: float = 5e-3,
    g_smear: Optional[np.ndarray] = None,
    xgrid: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Construct the full conductor Green's function with lead self-energies included.

    Parameters
    ----------
    `energy` : float
        Real energy value to evaluate the Green's function.
    `h_c` : np.ndarray
        Hamiltonian of the central conductor region (dim x dim).
    `s_c` : np.ndarray
        Overlap matrix of the conductor region (dim x dim).
    `sigma_l` : np.ndarray
        Self-energy matrix from the left lead (dim x dim).
    `sigma_r` : np.ndarray
        Self-energy matrix from the right lead (dim x dim).
    `surface` : bool
        If True, exclude the right self-energy (projected surface Green's function).
    `smearing_type` : str
        Smearing method: 'lorentzian', 'none', or 'numerical'.
    `delta` : float
        Smearing parameter for imaginary broadening.
    `delta_ratio` : float
        Used for 'none' smearing: `delta_eff = delta * delta_ratio`.
    `g_smear` : np.ndarray, optional
        Precomputed smeared Green’s function values (only for numerical smearing).
    `xgrid` : np.ndarray, optional
        Energy grid corresponding to `g_smear`.

    Returns
    -------
    `g_c` : np.ndarray
        Green's function of the conductor region (dim x dim).

    Notes
    -----
    Evaluates the full retarded Green's function:

        G_C(E) = [E·S - H - Σ_L - Σ_R]⁻¹     (if surface is False)
        G_C(E) = [E·S - H - Σ_L]⁻¹          (if surface is True)

    using smearing to regularize the inversion.
    """
    from gzero_maker import compute_non_interacting_gf

    sigma_total = sigma_l if surface else sigma_l + sigma_r

    try:
        g_c = compute_non_interacting_gf(
            energy,
            h_c + sigma_total,
            s_c,
            smearing_type=smearing_type,
            delta=delta,
            delta_ratio=delta_ratio,
            g_smear=g_smear,
            xgrid=xgrid,
            calc="direct",
        )
    except np.linalg.LinAlgError as e:
        raise RuntimeError("Failed to invert conductor Green's function matrix.") from e

    return g_c
