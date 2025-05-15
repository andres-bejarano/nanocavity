import numpy as np
import scipy.special as sp


def FC(q1, q2, g, method="Koch"):
    """
    This function calculates the Franck-Condon matrix element which is:
        $\langle q1 | \exp[-g (\op{b}^\dagger - \op{b})] | q2 \rangle$.
        q1 and q2 are bosonic states and \op{b} is the bosonic annihilation operator

    See Eq. B(3) in:
    J. Koch, F. von Oppen, Y. Oreg and E. Sela
    Thermopower of single-molecule devices
    Phys. Rev. B 70, 195107 (2004)

    As another reference see Eq. A(2) in:
    A. Yar, A. Donarini, S. Koller and M. Grifoni
    Dynamical symmetry breaking in vibration-assisted transport through nanostructures
    Phys. Rev. B 84, 115432 (2011)


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
    err_string = (
        "q1 and q2 must be nonnegative integers or lists of nonnegative integers."
    )
    for q in (q1, q2):
        if isinstance(q, (list, np.ndarray)):
            q = np.array(q)
            if q.dtype != np.int64:
                raise TypeError(err_string)
            elif (q < 0).any():
                raise ValueError(err_string)
        elif not isinstance(q, int):
            raise TypeError(err_string)
        elif q < 0:
            raise ValueError(err_string)

    if isinstance(g, int):
        g = float(g)

    n, m = np.meshgrid(q2, q1, indexing="ij")
    q = np.minimum(n, m)
    Q = np.maximum(n, m)
    fq = sp.factorial(q)
    fQ = sp.factorial(Q)
    L = sp.eval_genlaguerre(q, Q - q, g**2)
    common = np.sqrt(fq / fQ) * np.exp(-(g**2) / 2) * L

    if method == "Koch":
        assert np.imag(g) == 0  # this formula assumes a real valued coupling
        H = np.ones(n.shape)
        exp_arr = m - n
        H[exp_arr > 0] = (-1) ** exp_arr[exp_arr > 0]
        M = H * g ** (Q - q) * common
    elif method == "Yar":
        pre = np.heaviside(n - m, 0.5) * g ** (n - m) + np.heaviside(m - n, 0.5) * (
            -np.conj(g)
        ) ** (m - n)
        M = pre * common
    else:
        raise Exception("Need to provide a valid method.")

    if np.isscalar(q1) and np.isscalar(q2):
        return M[0].item()
    elif np.isscalar(q1) or np.isscalar(q2):
        return M.ravel()
    return M
