import numpy as np
import scipy.special as sp
from math import factorial


def FC_factor(nr_bosons, g):
    """Function to calculate the squared Franck-Condon Factors
    Parameters:
    --------------
        nr_bosons: int
        number of considered bosons in the system
        g: float
        coupling strength
    Returns:
    --------------
        FC: np.array
        Matrix containing the Franck-Condon Factors
        Probably calculated according to:
            Koch, von Oppen, Andreev, PRB 74, 205438 (2006),
            Theory of the Franck-Condon blockade regime
    """
    a = np.arange(nr_bosons + 1)
    nr_bosons, m = np.meshgrid(a, a)
    q = np.minimum(nr_bosons, m)
    Q = np.maximum(nr_bosons, m)
    fq = sp.factorial(q)
    fQ = sp.factorial(Q)
    L = sp.eval_genlaguerre(q, Q - q, g**2)
    M2 = fq / fQ * np.exp(-(g**2)) * (g ** (Q - q) * L) ** 2
    return M2
