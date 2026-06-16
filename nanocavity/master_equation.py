import numpy as np
from scipy.linalg import eig, expm
from secondquant.operator import Operator

import nanocavity.distributions as ndist
import nanocavity.operators as no


def reduced_population(nr_op, rho, values=[]):
    """
    Function to extract the reduced populations or one specific population of a density matrix
    Parameters:
        --------
        nr_op: list of secondquant operators or 2d-arrays
            Number operator used to specify which indices to extract
        rho: nd-array (2 at minimum)
            Density matrix from which the reduced populations should be extracted.
            The last two dimensions correspond to a density matrix.
            Therefore, one can for example pass a 4d array where the first two indices
            represent a loop over the bias
        values: list of integers
            If a specific population should be returned this specifies the state
            whose population should be returned
            Needs to be of the same length as nr_op
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
                    # Converting the matrix to a secondquant opeartor
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
    norm = np.einsum("ai,ai->i", vl.conj(), vr) ** 0.5
    vl /= norm.conj()
    vr /= norm
    return El, vl, vr


def stationary(
    L, method="eig", row=0, scale=1e-12, tol=1e-16, check=True, verbose=True
):
    """Solves for the steady state of the QME

    Parameters
    ----------

    method : {"eig", "solve"}
        whether to use np.linalg.eig or np.linalg.solve
    row : int, optional
        sets the row which gets overwritten
    scale : float, optional
        scale factor for the diagonal entries in the modified Liouvillian
    tol : float, optional
        clip value applied to eigenvalues of rho_st (clipping disabled if set to False)
    check : bool, optional
        whether to perform sanity checks of the computed density matrix
    verbose : bool
        print smallest and largest eigenvalues of rho_st (only if check=True)

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

    if tol:
        # ensure hermitian
        rho = (rho + rho.conj().T) / 2
        evals, evecs = np.linalg.eigh(rho)
        # clip eigenvalues >= tol
        evals_clipped = np.clip(evals, tol, None)
        rho = (evecs * evals_clipped) @ evecs.conj().T

    # normalize
    rho /= np.trace(rho)

    if check:
        # ensure a physically meaningful density matrix
        assert np.isclose(np.trace(rho), 1.0)
        # check solution as stationary
        drho = L @ rho.reshape(d**2)
        assert np.allclose(drho, 0)
        # check hermiticity
        assert np.allclose(rho, rho.conj().T)
        # check positivity
        evals = np.linalg.eigvalsh(rho)
        if verbose:
            print(
                f"Smallest [largest] eigenvalue of rho_st: {np.min(evals)} [{np.max(evals)}]"
            )
        assert np.all(evals >= -np.abs(tol))  # tolerate small negative numbers

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
    A, L, B, rho_st=None, avgA=False, avgB=False, verbose=True, cutoff=0
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
    avgA : bool
        whether to return the expectation value <A>
    avgB : bool
        whether to return the expectation value <B>
    verbose : bool
        print all M_k coefficients and corresponding complex eigenvalues E_k above cutoff
    cutoff : float
        the precision of the values to be considered in the M coefficients

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

    # keep only data meeting the cutoff criterium
    idx = np.where(np.abs(M) >= cutoff)[0]
    M = M[idx]
    E = E[idx]

    if verbose:
        print(f"{'k':>4} {'Re(Mk)':>16} {'Im(Mk)':>16} {'Re(Ek)':>16} {'Im(Ek)':>16}")
        for k, Ek in enumerate(E):
            print(
                f"{k:4} {M[k].real:16.8e} {M[k].imag:16.8e} {Ek.real:16.8e} {Ek.imag:16.8e}"
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
    A, L, B, time_delay, cutoff=0, verbose=True, ret_data=False, rho_st=None
):

    M, E = regression_theorem(
        A, L, B, rho_st, verbose=verbose, cutoff=cutoff
    )

    # correlation function S is generally complex
    time_delay = _toarray(time_delay)
    S = np.zeros(len(time_delay), dtype=np.complex128)

    for k, Ek in enumerate(E):
        S += M[k] * np.exp(Ek * time_delay)

    if ret_data:
        # S, weights, eigenvalues
        return S, M, E

    return S


def spectrum_resolvent(L, A, frequency, rho_st=None, zero_mode_tol=None):
    """Computes the frequency spectrum S(frequency) through calculation
    of the Liouvillian resolvent per frequency value.

    Parameters
    ----------
    L : ndarray
        Liouvillian superoperator
    A : ndarray
        Annihilation operator
    frequency : float or ndarray
        Frequencies (energies) to be evaluated
    rho_st : ndarray, optional
        steady-state density matrix
    remove_zero_tol : float, optional
        tolerance for eigenmodes of L to be removed

    Returns
    -------
    S : ndarray
        Spectrum of A
    """

    if rho_st is None:
        # we will rely on the default method for obtaining rho_st
        rho_st = stationary(L)

    ev, V = np.linalg.eig(L)
    Vinv = np.linalg.inv(V)

    modes_removed = 0
    if zero_mode_tol is not None:
        # Remove modes below tolerance
        mask = np.abs(ev) > zero_mode_tol
        ev = ev[mask]
        V = V[:, mask]
        Vinv = Vinv[mask, :]
        modes_removed = np.count_nonzero(~mask)
    print(f"spectrum_resolvent: Removed {modes_removed} eigenpair(s) using tol={zero_mode_tol}")
    if modes_removed != 1:
        print("  --> WARNING: theoretically exactly one mode of L should be removed!")

    Arho_vec = (A @ rho_st).toarray().reshape(-1)
    Adag_vec = A.toarray().conj().reshape(-1)

    frequency = _toarray(frequency)
    R = 1 / (1j * frequency.reshape(1, -1) - ev.reshape(-1, 1))
    S = np.einsum("i,iw,i->w", Adag_vec @ V, R, Vinv @ Arho_vec)

    return S.real / np.pi


def spectrum(L, A, frequency, cutoff=0, verbose=True, ret_data=False, rho_st=None):
    """Uses the regression theorem to compute the first-order correlation function
    :math:`Re \\int_0^\\infty < A^\\dagger(\\tau) A(0) > e^{-i \\omega \\tau} d\\tau`.

    Parameters
    ----------
    L : ndarray
        Liouvillian superoperator
    A : ndarray
        Annihilation operator
    frequency : float or ndarray
        Frequencies (energies) to be evaluated
    cutoff : float
        the precision of the values to be considered in the M coefficients coming from regression_theorem
    verbose : bool
        print all M_k coefficients and corresponding complex eigenvalues E_k above cutoff
    ret_data : bool
        whether to return also G1, M_k, and E_k
    rho_st : ndarray
        steady-state density matrix

    Returns
    -------
    I : ndarray
        computed spectrum on frequency (energy) grid
    M : ndarray
        coefficients M_k
    E : ndarray
        eigenvalues E_k of L
    """
    if verbose:
        print(f"Computing spectrum with cutoff={cutoff}")
    M, E = regression_theorem(
        A.d, L, A, rho_st, verbose=verbose, cutoff=cutoff
    )

    frequency = _toarray(frequency)
    I = np.zeros(len(frequency), dtype=np.float64)

    for k, Ek in enumerate(E):
        I += M[k].real * ndist.lorentzian(frequency - Ek.imag, -2 * Ek.real)

    if ret_data:
        # spectrum, weights, eigenvalues
        return I, M, E

    return I


def g2(
    L,
    A,
    time_delay,
    method="eigen",
    cutoff=0,
    verbose=True,
    ret_data=False,
    rho_st=None,
):
    """Uses the quantum regression theorem to compute the second-order correlation function
    :math:`\\langle A^\\dagger(0) A^\\dagger(\\tau) A(\\tau) A(0)\\rangle / \\langle A^\\dagger A \\rangle^2`.
    The method 'eigen' expands in terms of Liouvillian eigenvalues :math:`E_k`, such that
    :math:`G^2 = \\sum_k M_k e^{E_k \\tau}` with :math:`M_k = (w0, J v_k)(w_k, J rho_{st})`
    where :math:`J = A \\rho A^\\dagger` is a kind of jump superoperator.
    The method 'direct' traces all involved operators :math:`G^2 = Tr(J e^{L \\tau} J rho_{st})`.
    The function is returned normalized :math:`G1 = (w0, J rho_{st})`, then :math:`g2 = G2 / G1^2`

    Parameters
    ----------
    L : ndarray
        Liouvillian superoperator
    A : ndarray
        Annihilation operator
    time_delay : float or ndarray
        Time delay
    method : str
        'eigen' to expand in eigenvalues of L or 'direct' to calculate the trace of all the operators involved
    cutoff : float
        the precision of the values to be considered in the 'eigen' method for M coefficients coming from regression_theorem
    verbose : bool
        print all M_k coefficients and corresponding complex eigenvalues E_k above cutoff
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

    time_delay = _toarray(time_delay)
    G2 = np.zeros(len(time_delay), dtype=np.complex128)

    J = no.jump(A)

    if verbose:
        print(f"Computing g2 with cutoff={cutoff} and method={method}")

    if method == "eigen":
        M, E, G1 = regression_theorem(
            J, L, J, rho_st, verbose=verbose, avgA=True, cutoff=cutoff
        )

        for k, Ek in enumerate(E):
            G2 += M[k] * np.exp(Ek * time_delay)

    elif method == "direct":

        if rho_st is None:
            rho_st = stationary(L)
        dim = rho_st.shape[0]
        rho_st = rho_st.reshape(dim**2)
        w0 = np.eye(dim).reshape(dim**2)
        A = w0 @ J
        C = J @ rho_st
        for k, t in enumerate(time_delay):
            G2[k] += A @ expm(L * t) @ C
        G1 = A @ rho_st
        M, E = None, None

    # taking real parts BEFORE computing normalized g2:
    G1 = G1.real
    G2 = G2.real
    g2 = G2 / G1**2

    if ret_data:
        return g2, G1, M, E

    return g2


def g2_zero(A, rho_st, verbose=True):
    """Computes the same-time second-order correlation function
    from the exact expression g2(0) = (<n^2> - <n>) / <n>^2 where
    n = A.d * A.
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
