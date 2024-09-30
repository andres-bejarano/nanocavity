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


def _operator2super(A, dim):
    if isinstance(A, Operator):
        A = A.toarray()
    if A.shape[0] == dim:
        Id = np.eye(dim)
        A = np.kron(A, Id)
    return A


def regression_theorem(
    A, L, B, rho_st=None, sort=True, avgA=False, avgB=False, verbose=True
):
    """Uses the quantum regression theorem to compute two-operator coefficients and L-eigenvalues

    Parameters
    ----------
    A : {ndarray, Operator}
        operator or superoperator
    L : ndarray
        Liouvillian superoperator
    B : {ndarray, Operator}
        operator or superoperator
    rho_st : ndarray
        steady-state density matrix
    sort : bool
        whether to sort coefficients according to magnitude |M_k|
    avgA : bool
        whether to return the expectation value <A>
    avgB : bool
        whether to return the expectation value <B>
    verbose : bool
        print 10 dominant coefficients |M_k| and corresponding complex eigenvalues E_k

    Returns
    -------
    Coefficients, Eigenvalues, (<A>), (<B>)
    """

    if rho_st is None:
        # we will rely on the default method for obtaining rho_st
        rho_st = stationary(L)

    dim = rho_st.shape[0]
    A = _operator2super(A, dim)
    B = _operator2super(B, dim)
    w0 = np.eye(dim).reshape(dim**2)
    rho_st = rho_st.reshape(dim**2)
    E, vl, vr = eig_norm(L)

    # intermediate quantities, potentially reused
    w0A = w0 @ A
    Brho = B @ rho_st

    # coefficients M_k = <ALB>_k
    M = (w0A @ vr) * (vl.conj().T @ Brho)

    if sort:
        # sort descending
        idx = np.argsort(-np.abs(M))
        M = M[idx]
        E = E[idx]

    if verbose:
        # print up to 10 leading contributions according to magnitude of M[k]
        idx = np.argsort(-np.abs(M))[:10]
        print(f"\n{'k':>4} {'Re(Mk)':>12} {'Im(Mk)':>12} {'Re(Ek)':>12} {'Im(Ek)':>12}")
        for k in idx:
            print(
                f"{k:4} {M[k].real:12.6f} {M[k].imag:12.6f} {E[k].real:12.6f} {E[k].imag:12.6f}"
            )

    # compute also expectation values?
    if avgA and avgB:
        return M, E, w0A @ rho_st, w0 @ Brho
    if avgA:
        return M, E, w0A @ rho_st
    if avgB:
        return M, E, w0 @ Brho

    return M, E


def correlation_AB(
    A, L, B, tlist, cutoff=1e-12, verbose=True, ret_data=False, rho_st=None
):

    M, E = regression_theorem(A, L, B, rho_st, verbose=verbose)

    # correlation function S is generally complex
    tlist = _toarray(tlist)
    S = np.zeros(len(tlist), dtype=np.complex128)

    for k in range(len(E)):
        if abs(M[k]) > cutoff:
            S += M[k] * np.exp(E[k] * tlist)

    if ret_data:
        # S, weights, eigenvalues
        return S, M, E

    return S


def spectrum(L, a, wlist, cutoff=1e-12, verbose=True, ret_data=False, rho_st=None):

    M, E = regression_theorem(a.d, L, a, rho_st, verbose=verbose, sort=False)

    # spectrum is a real quantity, so we can skip the imaginary part
    M = M.real
    wlist = _toarray(wlist)
    I = np.zeros(len(wlist), dtype=np.float64)

    for k in range(len(E)):
        if M[k] > cutoff:
            I += M[k] * ndist.lorentzian(wlist - E[k].imag, -2 * E[k].real)

    if ret_data:
        # spectrum, weights, eigenvalues
        idx = np.argsort(-M)
        return I, M[idx], E[idx]

    return I


def g2(L, J, tlist, cutoff=1e-12, verbose=True, ret_data=False, rho_st=None):

    M, E, G1 = regression_theorem(J, L, J, rho_st, verbose=verbose, avgA=True)

    M /= G1**2

    tlist = _toarray(tlist)
    g2 = np.zeros(len(tlist), dtype=np.complex128)

    for k in range(len(E)):
        if abs(M[k]) > cutoff:
            g2 += M[k] * np.exp(E[k] * tlist)

    if ret_data:
        # g2, weights, eigenvalues
        return g2.real, M, E

    return g2.real
