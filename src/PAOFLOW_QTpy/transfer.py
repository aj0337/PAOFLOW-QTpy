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
    """
    Iteratively compute surface transfer matrices using the Sancho-Rubio method.

    Parameters
    ----------
    `h_eff` : np.ndarray
        Effective Hamiltonian block H_00 (ndim x ndim).
    `s_eff` : np.ndarray
        Overlap matrix S_00 (ndim x ndim).
    `t_coupling` : np.ndarray
        Coupling matrix H_01 (ndim x ndim).
    `delta` : float
        Small imaginary part added to stabilize inversion.
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
        Transfer matrix T (ndim x ndim).
    `tott` : np.ndarray
        Conjugate transfer matrix T† (ndim x ndim).
    `niter` : int
        Number of iterations used.
    """
    ndim = h_eff.shape[0]
    A = h_eff + 1j * delta * s_eff

    try:
        t11 = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        if verbose:
            logger.warning("Initial inversion failed: singular matrix.")
        if fail_counter is not None:
            fail_counter["nfail"] = fail_counter.get("nfail", 0) + 1
            if fail_counter["nfail"] > fail_limit:
                raise RuntimeError("Too many failures in transfer matrix convergence.")
        return (
            np.zeros((ndim, ndim), dtype=np.complex128),
            np.zeros((ndim, ndim), dtype=np.complex128),
            0,
        )

    tau = t11 @ t_coupling.conj()
    taut = t11 @ t_coupling

    tot = tau.copy()
    tott = taut.copy()
    tsum = taut.copy()
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
                    f"Singular matrix at iteration {m}; discarding energy point."
                )
            if fail_counter is not None:
                fail_counter["nfail"] = fail_counter.get("nfail", 0) + 1
                if fail_counter["nfail"] > fail_limit:
                    raise RuntimeError(
                        "Too many failures in transfer matrix convergence."
                    )
            return (
                np.zeros((ndim, ndim), dtype=np.complex128),
                np.zeros((ndim, ndim), dtype=np.complex128),
                m,
            )

        t11 = tau @ tau
        t12 = taut @ taut
        tau_next = s2 @ t11
        taut_next = s2 @ t12

        tot += tsum @ tau_next
        tsum = tsum @ taut_next

        tott += tsumt @ taut_next
        tsumt = tsumt @ tau_next

        tau = tau_next
        taut = taut_next

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
    return (
        np.zeros((ndim, ndim), dtype=np.complex128),
        np.zeros((ndim, ndim), dtype=np.complex128),
        niterx,
    )
