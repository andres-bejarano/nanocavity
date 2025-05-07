import numpy as np
import scipy.special as sp


def FC(a, b, g):
    """
    Franck-Condon matrix element

    See Eq. A(2) in:
    A. Yar, A. Donarini, S. Koller and M. Grifoni
    Dynamical symmetry breaking in vibration-assisted transport through nanostructures
    Phys. Rev. B 84, 115432 (2011)

    See Eq. B(3) in:
    J. Koch, F. von Oppen, Y. Oreg and E. Sela
    Thermopower of single-molecule devices
    Phys. Rev. B 70, 195107 (2004)

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

    H = np.ones(n.shape)
    exp_arr = m - n
    H[exp_arr > 0] = (-1) ** exp_arr[exp_arr > 0]
    M = H * np.sqrt(fq / fQ) * np.exp(-(g**2) / 2) * g ** (Q - q) * L

    if np.isscalar(a) and np.isscalar(b):
        return M[0].item()
    elif np.isscalar(a) or np.isscalar(b):
        return M.ravel()
    return M
