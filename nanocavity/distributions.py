import numpy as np

#fermi-dirac distribution
def fermi(E, T=10e-1, mu=0):
    return 1. / (np.exp((E - mu) / T) + 1.)

#bose-einstein distribution
def bose(E, T=10e-1, mu=0):
    return 1. / (np.exp((E - mu) / T) - 1.)
