import numpy as np
import nanocavity.distributions as ndist
from scipy.linalg import eig
from secondquant.operator import Operator


def eig_norm(L):
    El, vl, vr = eig(L, left=True)
    norm = np.einsum("ai,ai->i", vl.conj(), vr) ** -0.5
    vl *= norm.conj()
    vr *= norm
    return El, vl, vr


def stationary(L, method="eig", row=0, scale=1e-12, check=True, tol=-1e8):
    """Solves for the steady state of the QME

    Parameters
    ----------

    method : {"eig", "solve"}
        whether to use np.linalg.eig or np.linalg.solve
    row : int, optional
        sets the row which gets overwritten
    scale : float, optional
        scale factor for the diagonal entries in the modified Liouvillian

    Returns
    -------
    density matrix (2D array)
    """
    d = int(np.sqrt(L.shape[0]))
    if method == "eig":
        E, V = np.linalg.eig(L)
        # find the zero-eigenvalue mode index
        idx0 = np.argmin(np.abs(E))
        rho = V[:, idx0]
    elif method == "solve":
        L0 = L.copy()
        b = scale * np.eye(d).reshape(d**2)
        # find corresponding superindex row
        sl = row * (d + 1)
        L0[sl] = b
        rho = np.linalg.solve(L0, b)
    rho = rho.reshape(d, d)
    rho /= np.trace(rho)
    if check:
        # ensure a physically meaningful density matrix
        assert np.isclose(np.trace(rho), 1.0)
        # check solution as stationary
        drho = L @ rho.reshape(d**2)
        assert np.isclose(np.sum(drho), 0)
        # check hermiticity
        assert np.allclose(rho, rho.conj().T)
        # check positivity
        evals = np.linalg.eigvals(rho)
        assert np.all(evals >= tol)  # tolerate small negative numbers
    return rho


def current(J, L):
    # w/v left/right eigenvectors
    El, vl, vr = eig_norm(L)
    index = np.argmin(np.abs(El))
    return vl[:, index].conj() @ J @ vr[:, index]


def _toarray(x):
    if np.isscalar(x):
        x = [x]
    return np.array(x)

def correlation_AB(L, A, B, tlist, cutoff=1e-12):
    tlist = _toarray(tlist)

    if isinstance(A, Operator):
        A = A.toarray()

    if isinstance(B, Operator):
        B = B.toarray()

    dim = A.shape[0]
    Id = np.eye(dim)

    A = np.kron(A, Id)
    B = np.kron(B, Id)

    w0 = np.eye(dim).reshape(dim**2)
    rho_st = stationary(L).reshape(dim**2)
    El, vl, vr = eig_norm(L)

    S = np.zeros(len(tlist), dtype=np.complex128)

    for k in range(0, len(El)):
        Ak = w0 @ A @ vr[:, k]
        Bk = vl[:, k].conj() @ B @ rho_st
        if abs(Ak) > cutoff:
            S += Ak * Bk * np.exp(El[k] * tlist)
    return S


def spectrum(L, a, wlist, cutoff=1e-12, verbose=True):
    wlist = _toarray(wlist)

    if isinstance(a, Operator):
        a = a.toarray()

    dim = a.shape[0]
    Id = np.eye(dim)

    Ad = np.kron(a.conj().T, Id)
    A = np.kron(a, Id)

    w0 = np.eye(dim).reshape(dim**2)
    rho_st = stationary(L).reshape(dim**2)
    El, vl, vr = eig_norm(L)

    I = np.zeros(len(wlist), dtype=np.complex128)

    if verbose:
        print(
            f"{'k':4s} {'Ak.abs':12s} {'Bk.abs':12s} {'Mk.abs':12s} {'El[k].real':12s} {'El[k].imag':12s}"
        )
    for k in range(len(El)):
        Ak = w0 @ Ad @ vr[:, k]
        Bk = vl[:, k].conj() @ A @ rho_st
        if abs(Ak) > cutoff:
            if verbose:
                print(
                    f"{k:4} {np.abs(Ak):12.6f} {np.abs(Bk):12.6f} {np.abs(Ak * Bk):12.6f} {El[k].real:12.6f} {El[k].imag:12.6f}"
                )
            Dist = ndist.lorentzian(wlist - El[k].imag, -2 * El[k].real)
            I += Ak * Bk * Dist
    return I.real


def g2(L, J, tlist, cutoff=1e-12):
    tlist = _toarray(tlist)

    if isinstance(J, Operator):
        J = J.toarray()

    dim = J.shape[0]

    w0 = np.eye(int(np.sqrt(dim))).reshape(1, dim)
    rho_st = stationary(L).reshape(dim)
    El, vl, vr = eig_norm(L)

    G1 = w0 @ J @ rho_st
    G2 = np.zeros(len(tlist), dtype=np.complex128)

    for k in range(0, len(El)):
        Ak = w0 @ J @ vr[:, k]
        Bk = vl[:, k].conj() @ J @ rho_st
        if abs(Ak) > cutoff:
            G2 += Ak * Bk * np.exp(El[k] * tlist)
    g2 = G2 / G1**2
    return g2.real
