import logging
from typing import Tuple
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def compute_surface_transfer_matrices(
    h_eff: np.ndarray,
    s_eff: np.ndarray,
    t_coupling: np.ndarray,
    delta: float = 1e-5,
    niterx: int = 200,
    transfer_thr: float = 1e-7,
    fail_counter: dict = None,
    fail_limit: int = 5,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Iteratively compute surface transfer matrices using the Sancho-Rubio method.

    Parameters
    ----------
    `h_eff` : np.ndarray
        Effective Hamiltonian block H_00 (dim x dim).
    `s_eff` : np.ndarray
        Overlap matrix S_00 (dim x dim).
    `t_coupling` : np.ndarray
        Coupling matrix H_01 (dim x dim).
    `delta` : float
        Small imaginary part added to stabilize inversion (default: 1e-8).
    `niterx` : int
        Maximum number of iterations.
    `transfer_thr` : float
        Convergence threshold.
    `fail_counter` : dict
        Mutable dict to count number of convergence failures.
    `fail_limit` : int
        Maximum number of failures before raising an exception.
    `verbose` : bool
        If True, enables logging output.

    Returns
    -------
    `tot` : np.ndarray
        Transfer matrix T (dim x dim).
    `tott` : np.ndarray
        Conjugate transfer matrix T† (dim x dim).
    `niter` : int
        Number of iterations used.

    Notes
    -----
    Given transfer matrices `T` and `T†`, computes the surface or bulk Green's function
    as follows:

    - If `igreen == 1`: `G = [E⋅S - H - H_01 ⋅ T]⁻¹` (right surface)
    - If `igreen == -1`: `G = [E⋅S - H - H_01† ⋅ T†]⁻¹` (left surface)
    - If `igreen == 0`: `G = [E⋅S - H - H_01 ⋅ T - H_01† ⋅ T†]⁻¹` (bulk)

    """
    ndim = h_eff.shape[0]
    z_shift = 1j * delta * s_eff
    A = h_eff + z_shift

    try:
        t11 = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        if verbose:
            logger.warning(
                "Initial inversion failed: singular matrix in transfer matrix setup."
            )
        if fail_counter is not None:
            fail_counter["nfail"] = fail_counter.get("nfail", 0) + 1
            if fail_counter["nfail"] > fail_limit:
                raise RuntimeError("Too many failures in transfer matrix convergence.")
        return (
            np.zeros((ndim, ndim), dtype=np.complex128),
            np.zeros((ndim, ndim), dtype=np.complex128),
            0,
        )

    tau = t11 @ t_coupling.conj().T
    taut = t11 @ t_coupling
    tot = tau.copy()
    tsum = taut.copy()
    tott = taut.copy()
    tsumt = tau.copy()

    for m in range(1, niterx + 1):
        t11_new = tau @ taut
        t12_new = taut @ tau
        s1 = -(t11_new + t12_new)
        np.fill_diagonal(s1, 1.0 + np.diag(s1))
        try:
            s2 = np.linalg.inv(s1)
        except np.linalg.LinAlgError:
            if verbose:
                logger.warning(
                    f"Singular matrix encountered at iteration {m}; discarding energy point."
                )
            if fail_counter is not None:
                fail_counter["nfail"] = fail_counter.get("nfail", 0) + 1
                if fail_counter["nfail"] > fail_limit:
                    raise RuntimeError(
                        "Too many failures in transfer matrix convergence."
                    )
            return np.zeros_like(tot), np.zeros_like(tott), m

        t11_next = s2 @ (tau @ tau)
        t12_next = s2 @ (taut @ taut)
        tot += tsum @ t11_next
        tsum = tsum @ t12_next
        tott += tsumt @ t12_next
        tsumt = tsumt @ t11_next
        tau = t11_next
        taut = t12_next

        conver = np.sum(np.abs(tau) ** 2).real
        conver2 = np.sum(np.abs(taut) ** 2).real
        if conver < transfer_thr and conver2 < transfer_thr:
            if verbose:
                logger.info(f"Transfer matrix converged after {m} iterations.")
            return tot, tott, m

    if verbose:
        logger.warning(f"Transfer matrix did not converge after {niterx} iterations.")
    if fail_counter is not None:
        fail_counter["nfail"] = fail_counter.get("nfail", 0) + 1
        if fail_counter["nfail"] > fail_limit:
            raise RuntimeError("Too many failures in transfer matrix convergence.")
    return np.zeros_like(tot), np.zeros_like(tott), niterx
