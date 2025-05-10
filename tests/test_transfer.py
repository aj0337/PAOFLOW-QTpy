import numpy as np
from PAOFLOW_QTpy.transfer import compute_surface_transfer_matrices


def test_compute_surface_transfer_matrices_converges():
    """Test that compute_surface_transfer_matrices converges for a simple Hermitian system."""
    dim = 3
    np.random.seed(0)

    # Hermitian H, positive-definite S, random T
    H = np.random.rand(dim, dim) + 1j * np.random.rand(dim, dim)
    H = (H + H.conj().T) / 2  # Make Hermitian

    S = np.eye(dim) * 1.1  # Slightly larger than identity to avoid singularities
    T = np.random.rand(dim, dim) + 1j * np.random.rand(dim, dim)

    fail_counter = {"nfail": 0}
    tot, tott, niter = compute_surface_transfer_matrices(
        H, S, T, delta=1e-6, transfer_thr=1e-10, fail_counter=fail_counter, verbose=True
    )

    assert np.all(np.isfinite(tot)), "Transfer matrix contains non-finite entries."
    assert np.all(
        np.isfinite(tott)
    ), "Conjugate transfer matrix contains non-finite entries."
    assert niter < 100, "Convergence took too long or failed."
    assert fail_counter["nfail"] == 0, "Unexpected failure recorded in counter."


def test_compute_surface_transfer_matrices_failure():
    """Test that compute_surface_transfer_matrices handles singular matrix failure correctly."""
    dim = 3
    # Construct a singular H matrix (rank-deficient)
    H = np.zeros((dim, dim), dtype=np.complex128)
    S = np.eye(dim, dtype=np.complex128)
    T = np.eye(dim, dtype=np.complex128)

    fail_counter = {"nfail": 0}
    tot, tott, niter = compute_surface_transfer_matrices(
        H,
        S,
        T,
        delta=0.0,
        transfer_thr=1e-10,
        fail_counter=fail_counter,
        fail_limit=5,
        verbose=True,
    )

    assert np.allclose(tot, 0), "Transfer matrix should be zero after failure."
    assert np.allclose(
        tott, 0
    ), "Conjugate transfer matrix should be zero after failure."
    assert fail_counter["nfail"] == 1, "Failure counter should be incremented."
