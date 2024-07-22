import numpy as np
import pytest

from PAOFLOW_QTpy.smearing_T import smearing_func, smearing_init

def test_smearing_init():
    xmax = 10.0
    delta_ratio = 0.1
    smearing_type = "gaussian"

    g_smear = smearing_init(xmax, delta_ratio, smearing_type)

    assert g_smear.ndim == 1
    assert g_smear.shape[0] == 2 * int(2 * xmax / delta_ratio)
    assert g_smear.dtype == np.float64

    x_values = np.linspace(-xmax, xmax, 5)
    expected_values = [smearing_func(x, smearing_type) for x in x_values]

    assert np.allclose(g_smear[::5], expected_values, rtol=1e-5, atol=1e-8)
