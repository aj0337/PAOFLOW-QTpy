# import numpy as np
# import scipy.signal as sp
# from scipy.fft import fft, ifft
# from .smearing_base import fermidirac, gaussian, lorentzian, marzarivanderbilt, metpax


# def smearing_func(x, smearing_type):
#     if smearing_type == "gaussian":
#         gaussian(x)
#     elif smearing_type == "methfessel-paxton" or "mp":
#         metpax(x)
#     elif smearing_type == "marzari-vanderbilt" or "mv":
#         marzarivanderbilt(x)
#     elif smearing_type == "fermi-dirac" or "fd":
#         fermidirac(x)
#     elif smearing_type == "lorentzian":
#         lorentzian(x)


# def good_fft_order_1dz(nfft):
#     raise NotImplementedError


# def smearing_init(xmax, delta, delta_ratio, smearing_type):
#     """
#     Initialize the smearing function.

#     Args:
#         xmax (float): Maximum value of the spatial grid.
#         delta_ratio (float): Ratio of the smearing width to the spatial grid spacing.
#         smearing_type (str): Type of smearing function (e.g., "gaussian", "exponential").

#     Returns:
#         g_smear (numpy.ndarray): Smearing function.
#     """

#     if delta_ratio < 0:
#         raise ValueError("delta_ratio must be non-negative")
#     if delta_ratio > 1:
#         raise ValueError("delta_ratio must be less than or equal to 1")

#     nx = 2 * int(2 * xmax / delta_ratio)
#     dx = 2 * xmax / nx

#     xgrid = np.linspace(-xmax, xmax, nx)

#     eps_sx = 15.0  # Half of the width of the smearing function
#     eps_px = xmax + eps_sx
#     Tmax = xmax + 2 * eps_sx
#     nfft = 1 + int((Tmax / xmax) * nx)
#     # nfft = sp.good_size(nfft)

#     fft_grid = np.linspace(-xmax, xmax, nfft)

#     is_start = np.searchsorted(fft_grid, -eps_sx)
#     is_end = np.searchsorted(fft_grid, eps_sx)
#     ip_start = np.searchsorted(fft_grid, -eps_px)
#     ip_end = np.searchsorted(fft_grid, eps_px)

#     auxs_in = np.zeros(nfft, dtype=complex)
#     auxs_out = np.zeros(nfft, dtype=complex)

#     cost = 1 / delta

#     auxs_in[is_start:is_end] = cost * smearing_func(xgrid[is_start:is_end], smearing_type)

#     auxp_in = np.zeros(nfft, dtype=complex)
#     auxp_out = np.zeros(nfft, dtype=complex)

#     cost = 1

#     auxp_in[ip_start:ip_end] = cost / (xgrid[ip_start:ip_end] + 1j * delta_ratio)

#     wrapped = np.roll(auxs_in, -is_start)
#     auxs_in = wrapped

#     auxs_out = fft(auxs_in)
#     auxp_out = fft(auxp_in)

#     cost = 2 * Tmax
#     auxp_out *= cost * auxs_out

#     auxp_in = ifft(auxp_out)

#     ix_start = np.searchsorted(fft_grid, -xmax)
#     ix_end = ix_start + nx - 1
#     g_smear = np.real(auxp_in[ix_start:ix_end])

#     return g_smear
