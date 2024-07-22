import numpy as np


def gaussian(x):
    cost /= np.sqrt(np.pi)
    return cost * np.exp(-x**2)


def metpax(x):
    cost /= np.sqrt(np.pi)
    return cost * np.exp(-x**2) * (1.5 - x**2)


def marzarivanderbilt(x):
    cost /= np.sqrt(np.pi)
    return cost * np.exp(-(x - 1.0 / np.sqrt(2))**2) * (2.0 - np.sqrt(2) * x)


def fermidirac(x):
    cost /= 2.0
    return cost * 1.0 / (1.0 + np.cosh(x))


def lorentzian(x):
    cost /= np.pi
    return cost * 1.0 / (1.0 + x**2)
