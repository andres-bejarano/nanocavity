import numpy as np


# fermi-dirac distribution
def fermi_dirac(E, kT, mu=0):
    if not isinstance(E, np.ndarray):
        E = np.array(E)
    # 709 is approximately the largest number for which exp(x) won't overflow
    x = np.clip((E - mu) / kT, None, 709)
    return 1 / (np.exp(x) + 1)


# bose-einstein distribution
def bose_einstein(E, kT, mu=0):
    if not isinstance(E, np.ndarray):
        E = np.array(E)
    x = np.clip((E - mu) / kT, None, 709)
    safe_division = np.divide(1, np.expm1(x), where=(x != 0), out=np.zeros_like(x))
    return np.where(x == 0, np.inf, safe_division)


def bath_dist(E, kT, rate, bath, mu=0, eV=0):
    if bath == "bosonic":
        if rate == "in":

            def dist(E):
                return bose_einstein(-E, kT)

        elif rate == "out":

            def dist(E):
                return 1 + bose_einstein(E, kT)

    elif bath == "fermionic":
        if rate == "in":

            def dist(E):
                return fermi_dirac(-E, kT, mu)

        elif rate == "out":

            def dist(E):
                return 1 - fermi_dirac(E, kT, mu)

    elif bath == "leadtolead":
        if rate == "in":

            def dist(E):
                return float(Fermi_cb(eV + E, kT))

        elif rate == "out":

            def dist(E):
                return float(Fermi_cb(eV + E, kT))

    return dist


# lorentzian
def lorentzian(E, w, epsilon=0):
    denominator = w**2 / 4.0 + E**2
    denominator = np.where(denominator < epsilon, epsilon, denominator)
    return w / (2.0 * np.pi) / denominator


# fininte-band
def semi_circle(e, mu, w):
    e = np.array(e).reshape(-1, 1)
    mu = np.array(mu).reshape(1, -1)
    x = (e - mu) / w  # broadcasting
    x = np.clip(x, -1, 1)  # values outside the band width should be set to zero
    y = (1 - x**2) ** 0.5
    return np.squeeze(y)


# intregral of two fermi functions \int dy f_l(y)(1-f_r(y+x)) = x/(1-e^{-x/kT})
# it appears typically in Coulomb blockade
def Fermi_cb(E, kT):
    x = np.clip(E / kT, -709, None)
    safe_division = np.divide(x, -np.expm1(-x), where=(x != 0), out=np.zeros_like(x))
    return kT * np.where(x == 0, 1, safe_division)
