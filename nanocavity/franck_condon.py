import numpy as np
import scipy.special as sp


def FC2(a, b, g):
    """Squared Franck-Condon matrix elements M^2_{a,b}

    See Eq. (8) in:
    J. Koch, F. von Oppen, and A. V. Andreev
    Theory of the Franck-Condon blockade regime
    Phys. Rev. B 74, 205438 (2006)

    Parameters
    ----------
    a : int or array-like
       first index
    b : int or array-like
       second index
    g : float
       coupling strength

    Returns
    -------
    M2 : np.array or float
       Squared Franck-Condon matrix element(s)
    """

    n, m = np.meshgrid(a, b, indexing="ij")
    q = np.minimum(n, m)
    Q = np.maximum(n, m)
    fq = sp.factorial(q)
    fQ = sp.factorial(Q)
    L = sp.eval_genlaguerre(q, Q - q, g**2)
    M2 = fq / fQ * np.exp(-(g**2)) * (g ** (Q - q) * L) ** 2

    if np.isscalar(a) and np.isscalar(b):
        return M2[0].item()
    elif np.isscalar(a) or np.isscalar(b):
        return M2.ravel()
    return M2
