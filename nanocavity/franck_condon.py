import numpy as np
import scipy.special as sp


def FC(a, b, g):
    """
    Franck-Condon matrix element

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
    M : np.array or float
       Franck-Condon matrix element(s)
    """

    n, m = np.meshgrid(a, b, indexing="ij")
    q = np.minimum(n, m)
    Q = np.maximum(n, m)
    fq = sp.factorial(q)
    fQ = sp.factorial(Q)
    L = sp.eval_genlaguerre(q, Q - q, g**2)
    M = np.sqrt(fq / fQ) * np.exp(-(g**2) / 2) * g ** (Q - q) * L

    if np.isscalar(a) and np.isscalar(b):
        return M[0].item()
    elif np.isscalar(a) or np.isscalar(b):
        return M.ravel()
    return M


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
    M = FC(a, b, g)
    M2 = M * M

    return M2
