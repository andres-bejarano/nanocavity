import numpy as np

#fermi-dirac distribution
def fermi(E, T=10e-2, mu=0):
    return 1. / (np.exp((E - mu) / T) + 1.)

#bose-einstein distribution
def bose(E, T=10e-1, mu=0):
    return 1. / (np.exp((E - mu) / T) - 1.)


#fininte-band
def semi_circle(e, mu, w):
    e = np.array(e).reshape(-1, 1)
    mu = np.array(mu).reshape(1, -1)
    x = (e - mu) / w # broadcasting
    x = np.clip(x, -1, 1) # values outside the band width should be set to zero
    y = (1 - x ** 2) ** 0.5
    return np.squeeze(y)

