import numpy as np
import math

#fermi-dirac distribution
def fermi_dirac(E, kT=0.1, mu=0):
    return 1. / (np.exp((E - mu) / kT) + 1.)

#bose-einstein distribution
def bose_einstein(E, kT=0.1, mu=0):
    return 1. / (np.exp((E - mu) / kT) - 1.)

def bath_dist(E, kT, rate, bath, mu=0, eV=0):
    if bath == 'bosonic':
        if rate == 'in':
            def dist(E):
                return bose_einstein(-E, kT)
        elif rate == 'out':
            def dist(E):
                return 1 + bose_einstein(E, kT)
    elif bath == 'fermionic':
        if rate == 'in':
            def dist(E):
                return fermi_dirac(-E, kT, mu)
        elif rate == 'out':
            def dist(E):
                return 1 - fermi_dirac(E, kT, mu)
    elif bath == 'leadtolead':
        if rate == 'in':
            def dist(E):
                return float(Fermi_cb(eV+E, kT))
        elif rate == 'out':
            def dist(E):
                return float(Fermi_cb(eV+E, kT))
    return dist

#lorentzian
def lorentzian(E, w):
    return (1. / np.pi) * ((w / 2.) / ((w ** 2 / 4.) + E ** 2))

#fininte-band
def semi_circle(e, mu, w):
    e = np.array(e).reshape(-1, 1)
    mu = np.array(mu).reshape(1, -1)
    x = (e - mu) / w # broadcasting
    x = np.clip(x, -1, 1) # values outside the band width should be set to zero
    y = (1 - x ** 2) ** 0.5
    return np.squeeze(y)

#intregral of two fermi function \int dy f_l(y)(1-f_r(y+x)) = x/1-e^{x/kBT}
#it appears tipycally in coulomb blockade
def Fermi_cb(x, kT):
    return np.where(x == 0, kT, x/(1-np.exp(-x/kT)))
