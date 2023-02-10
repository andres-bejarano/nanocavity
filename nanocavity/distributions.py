import numpy as np

#fermi-dirac distribution
def fermi_dirac(E, kT=0.1, mu=0):
    return 1. / (np.exp((E - mu) / kT) + 1.)

#bose-einstein distribution
def bose_einstein(E, kT=0.1, mu=0):
    return 1. / (np.exp((E - mu) / kT) - 1.)

#lorentzian
def lorentz(E, w):
    return (1. / np.pi) * ((w / 2.) / ((w**2 / 4) + E**2))

#fininte-band
def semi_circle(e, mu, w):
    e = np.array(e).reshape(-1, 1)
    mu = np.array(mu).reshape(1, -1)
    x = (e - mu) / w # broadcasting
    x = np.clip(x, -1, 1) # values outside the band width should be set to zero
    y = (1 - x ** 2) ** 0.5
    return np.squeeze(y)

