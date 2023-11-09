import numpy as np


def smearing_init(smearing_type, x, delta):
    if smearing_type == 'gaussian':
        gaussian(x, delta)
    if smearing_type == 'methfessel-paxton' or 'mp':
        metpax(x, delta)
    if smearing_type == 'marzari-vanderbilt' or 'mv':
        marzarivanderbilt(x, delta)
    if smearing_type == 'fermi-dirac' or 'fd':
        fermidirac(x, delta)
    if smearing_type == 'lorentzian':
        lorentzian(x, delta)


def define_xgrid(xmax, delta_ratio):
    nx = 2 * int(2 * xmax / delta_ratio)
    dx = 2 * xmax / np.real(nx)
    nxindex = np.arange(0, nx, 1, dtype=int)
    xgrid = -np.real(nx / 2) * dx + nxindex * dx
    return xgrid


# TESTED define_xgrid outputs match


def define_fft_grid(xmax, delta_ratio):
    nx = 2 * int(2 * xmax / delta_ratio)
    dx = 2 * xmax / np.real(nx)
    eps_sx = 15.0
    Tmax = xmax + 2 * eps_sx
    nfft = 1 + int((Tmax / xmax) * nx)
    nfftindex = np.arange(0, nfft, 1, dtype=int)
    fft_grid = -np.real(nfft / 2) * dx + nfftindex * dx
    return nfft, fft_grid


# TESTED  define_fft_grid nfft values don't match because the fortran version
# does an additional good fft dimension calculation that is not done in the
# python version. Consequnetly, since fft_grid uses this nfft value,
# fft_grids don't match in the python vs fortran version. However the grids in
# both versions are equally spaced by 0.0025 eV in test01.


def gaussian(x, delta):
    return (np.exp(-((x) / delta)**2) / delta) / np.sqrt(np.pi)


def metpax(x, delta):
    from math import factorial

    from numpy.polynomial.hermite import hermval
    nh = 5
    coeff = np.zeros(2 * nh)
    coeff[0] = 1.
    for n in range(2, 2 * nh, 2):
        m = n // 2
        coeff[n] = (-1.)**m / (factorial(m) * (4.0**m) * np.sqrt(np.pi))
    return hermval(x / delta,
                   coeff) * np.exp(-(x / delta)**2) / (delta * np.sqrt(np.pi))


def marzarivanderbilt(x, delta):
    return np.exp(-(x - 1 / (np.sqrt(2)))**2) * (2.0 - np.sqrt(2) * x) / (
        delta * np.sqrt(np.pi))


def fermidirac(x, delta):
    return 1 / (delta * (1 + np.cosh(x)))


def lorentzian(x, delta):
    return 1 / (delta * (1 + x**2))


def define_input_pole(xmax, delta_ratio):
    from locate import locate_extrema

    eps_sx = 15.0
    eps_px = xmax + eps_sx
    nfft, fft_grid = define_fft_grid(xmax, delta_ratio)
    is_start = locate_extrema(nfft, fft_grid, -eps_px)
    is_end = locate_extrema(nfft, fft_grid, eps_px)
    auxp_in = 1 / (fft_grid[is_start:is_end] + 1j * delta_ratio)
    return auxp_in


def define_complex_fft_smear(xmax, delta_ratio):
    from locate import locate_extrema

    nfft, fft_grid = define_fft_grid(xmax, delta_ratio)
    is_extrema = locate_extrema(nfft, fft_grid, 0)
    if is_extrema < 0:
        is_extrema = is_extrema + 1
    return None


if __name__ == "__main__":
    from input import input_manager
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    input_dict = input_manager('test.yaml', comm)
    xmax = input_dict['xmax']
    delta_ratio = input_dict['delta_ratio']
    define_fft_grid(xmax, delta_ratio)
