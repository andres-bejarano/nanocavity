import numpy as np
import nanocavity.distributions as ndist
from scipy.linalg import eig, expm
from secondquant.operator import Operator


def reduced_populations(nr_op, rho, values=[]):
    """
    Function to extract the reduced populations of a density matrix
    Parameters:
        --------
        nr_op: Secondquant operator or 2d-array
            Number operator used to specify which indices to extract
        rho: nd-array (2 at minimum)
            Density matrix from which the reduced populations should be extracted.
            The last two dimensions correspond to a density matrix.
            Therefore, one can for example pass a 4d array where the first two indices
            represent a loop over the bias
    Returns:
        ------
        pops_reduced: (nd-1) array
            An array containing the reduced populations for each excitation of the type provided by nr_op
    """

    if not isinstance(nr_op, list):
        nr_op = [nr_op]
    for i, n in enumerate(nr_op):
        if not isinstance(n, Operator):
            if not isinstance(n, np.array()):
                raise TypeError(
                    "nr_op needs to be a secondquant operator or a 2d matrix"
                )
            else:
                if len(n.shape) == 2:
                    nr_op[i] = Operator(n)
                else:
                    raise TypeError(
                        "nr_op needs to be a secondquant operator or a 2d matrix"
                    )

    if len(nr_op) == 1:
        if not values:
            nr_pops = np.max(nr_op[0].toarray()) + 1
            pops_reduced = np.zeros(rho.shape[:-2] + (nr_pops,))
            for i in range(nr_pops):
                idx = nr_op[0].where(i)
                pops_reduced[..., i] = np.einsum("...k->...", rho[..., idx, idx].real)
        else:
            idx = nr_op[0].where(values[0])
            pops_reduced = np.einsum("...k->...", rho[..., idx, idx].real)
        return pops_reduced
    else:
        assert len(nr_op) == len(
            values
        ), "nr_op and values need to have the same length"
        idx = nr_op[0].where(values[0])
        for i, n in enumerate(nr_op[1:]):
            idx = np.intersect1d(idx, n.where(values[i + 1]))

        return np.sum(rho[..., idx, idx], axis=-1)


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


def average(A, rho_st):
    if A.shape == rho_st.shape:
        return np.trace(A @ rho_st)
    dim = rho_st.shape[0]
    A = _operator2super(A, dim)
    w0 = np.eye(dim).reshape(dim**2)
    rho_st = rho_st.reshape(dim**2)
    return w0 @ A @ rho_st


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


def correlation_AB(A, L, B, tlist, cutoff=0, verbose=True, ret_data=False, rho_st=None):

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


def spectrum(L, a, wlist, cutoff=0, verbose=True, ret_data=False, rho_st=None):

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


def g2(
    L, J, tlist, method="eigen", cutoff=0, verbose=True, ret_data=False, rho_st=None
):
    """Uses the quantum regression theorem to compute the second-order correlation functions.
    The method 'eigen' expands in terms of Liouvillian eigenvalues E_k, such that
    G2 = \sum_k M_k e^{E_k tau} with M_k = (w0, J v_k)(w_k, J rho_st).
    The method 'direct' traces all involved operatros G2 = Tr(J e^{L tau} J rho_st).
    The function is returned normalized G1 = (w0, J rho_st), then g2 = G2 / G1^2

    Parameters
    ----------
    J : ndarray
        Jump superoperator
    L : ndarray
        Liouvillian superoperator
    tlist : float or ndarray
        Time delay
    method : str
        'eigen' to expand in eigenvalues of L or 'direct' to calculate the trace of all the operators involved
    cutoff : float
        the precision of the values to be considered in the 'eigen' method
    verbose : bool
        print 10 dominant coefficients |M_k| and corresponding complex eigenvalues E_k
    ret_data : bool
        whether to return also G1, M_k, and E_k
    rho_st : ndarray
        steady-state density matrix

    Returns
    -------
    g2 : ndarray
        normalized second-order correlation function
    G1 : float
        first-order correlation function
    M : ndarray
        coefficients in the method 'eigen'
    E : ndarray
        eigenvalues of L in the method 'eigen'
    """

    tlist = _toarray(tlist)
    G2 = np.zeros(len(tlist), dtype=np.complex128)

    if method == "eigen":
        M, E, G1 = regression_theorem(J, L, J, rho_st, verbose=verbose, avgA=True)

        for k in range(len(E)):
            if abs(M[k]) > cutoff:
                G2 += M[k] * np.exp(E[k] * tlist)

    elif method == "direct":

        if rho_st is None:
            rho_st = stationary(L)
        dim = rho_st.shape[0]
        rho_st = rho_st.reshape(dim**2)
        w0 = np.eye(dim).reshape(dim**2)
        A = w0 @ J
        C = J @ rho_st
        for k, t in enumerate(tlist):
            G2[k] += A @ expm(L * t) @ C
        G1 = A @ rho_st
        M, E = None, None

    g2 = G2 / G1**2

    if ret_data:
        return g2.real, G1.real, M, E

    return g2.real


def g2_zero(A, rho_st, verbose=True):
    """Computes the same-time second-order correlation function
    from the exact expression g2(0) = (<n^2> - <n>) / <n>^2.
    """
    if isinstance(A, Operator):
        n = A.d * A
        n2 = n * n
        n = n.toarray()
        n2 = n2.toarray()
    else:
        n = A.conj().T @ A
        n2 = n @ n

    avg_n = average(n, rho_st).real
    avg_n2 = average(n2, rho_st).real

    g2 = (avg_n2 - avg_n) / (avg_n**2)

    if verbose:
        print(f"g2(0)      = {g2:.6f}")
        print(f" ... <n^2> = {avg_n2:.6e}")
        print(f" ... <n>   = {avg_n:.6e}")

    return g2
