import numpy as np
import numpy.linalg as la
from secondquant.operator import Operator

import nanocavity.distributions as ndist


def collapses(A_op, basis, kT, bath, rate, mu=0, cutoff=0):
    """
    Function to calculate the collapse operators which are needed to
    build a Liouvillian with secondquant operators
    Parameters:
        ----
        A_op: secondquantoperator
            annihilation operator
        basis: list
            (E, V) describing the basis for the collapse operators
        kT: float
            Temperature
        bath: string
            Either 'fermionic' or 'bosonic/bosonicX+'
        rate: float
            coupling strength
        mu: float
            chemical potential
        cutoff: float
            cutoff for the considered transition matrix elements

    Returns
    -------
        two lists of collapse operators for adding or removing particles
    """
    E, V = basis

    # Transition matrix elements between final (f) and initial (i) states
    # Thus the first index refers to the final state
    M_fi = A_op.inner(V)
    dim = A_op.shape[0]
    # Matrix of all energy differences in the system between final and initial
    # state
    E_fi = E.reshape(dim, 1) - E.reshape(1, dim)

    bath = bath.lower()
    if "bosonic" in bath:
        # This rate is for photon absorption, thus the final state must be
        # higher in energy than the initial one
        nb_fi_p = np.where(E_fi > 0, ndist.bose_einstein(E_fi, kT), 0)
        # This rate is for photon emission, thus the final state must be
        # lower in energy than the initial one
        nb_fi_m = np.where(E_fi < 0, 1 + ndist.bose_einstein(-E_fi, kT), 0)

    elif "fermionic" in bath:
        fd_fi_p = ndist.fermi_dirac(E_fi, kT, mu)
        fd_fi_m = 1 - ndist.fermi_dirac(-E_fi, kT, mu)

    if "x+" in bath and "bosonic" in bath:
        # following procedure in https://doi.org/10.1103/PRXQuantum.5.010312
        pos_bath_energy = np.where(E_fi < 0, -E_fi, 0)  # find increase of bath energy
        # construct operator X+ = sqrt(E_fi) * (a + a^\dagger)
        M_fi = (M_fi + M_fi.T) * np.sqrt(pos_bath_energy)
        print("building collapses with X+")

    cp, cm = [], []
    for f in range(dim):
        for i in range(dim):
            if abs(M_fi[f, i]) > cutoff:
                P = M_fi[f, i] * V[:, f].reshape(dim, 1) @ V[:, i].reshape(1, dim)
                if "bosonic" in bath:
                    cp.append(np.sqrt(rate * nb_fi_p.T[f, i]) * P.conj().T)
                    cm.append(np.sqrt(rate * nb_fi_m[f, i]) * P)

                elif bath == "fermionic":
                    cp.append(np.sqrt(rate * fd_fi_p.T[f, i]) * P.conj().T)
                    cm.append(np.sqrt(rate * fd_fi_m[f, i]) * P)
    return cp, cm


def jump(c_ops):
    if isinstance(c_ops, Operator):
        c_ops = c_ops.toarray()
    if isinstance(c_ops, np.ndarray):
        c_ops = [c_ops]
    if not isinstance(c_ops, list):
        raise TypeError("c_ops must be a single operator or a list of them")
    J = 0
    for c in c_ops:
        J += np.kron(c, c.conj())
    return J


def dissipator(c_ops, method="kron", diagonal_form=True):
    """
    Function to build the Dissipator with respect to given collapse operators

    Parameters:
    ----
    c_ops: list
        list of collapse operators
    method: string
        Can be "kron" or "einsum"
    diagonal_for: logical
        Keep the diagonal form of the Liouvillian or not.
        Defaults to True
    """
    if not isinstance(c_ops, list):
        raise TypeError("c_ops must be a list")
    dim = c_ops[0].shape[0]
    Id = np.eye(dim)
    # Look https://arxiv.org/pdf/1504.05266
    # Attention: The paper uses column stacking
    # This function uses row stacking
    D = 0
    for c1 in c_ops:
        c_ops2 = [c1] if diagonal_form else c_ops
        for c2 in c_ops2:
            cdc = c1.conj().T @ c2
            if method == "einsum":
                D += np.einsum("ik,jl->ijkl", c1, c2.conj())
                D -= 0.5 * np.einsum("ik,jl->ijkl", Id, cdc)
                D -= 0.5 * np.einsum("ik,jl->ijkl", cdc.conj(), Id)
            elif method == "kron":
                D += np.kron(c1, c2.conj())
                D -= 0.5 * np.kron(Id, cdc)
                D -= 0.5 * np.kron(cdc.conj(), Id)
    if method == "einsum":
        D = D.reshape((dim**2, dim**2))
    return D


def liouvillian(H, c_ops=None, method="kron", cond=True, diagonal_form=True):
    """
    Function calculating the Liouvillian for a central system coupled to baths
    Uses row stacking in the superspace.


    Parameters:
    ------------
    H: secondquant operator
        Hamiltonian describing the central system
    c_ops: list
        list of collapse operators
    method: string
        method on how to build the Liouvillian
        defaults to "kron"
    cond: logical
        Defaults to True
    diagonal_for: logical
        Keep the diagonal form of the Liouvillian or not.
        Defaults to True
        The diagonal form is the standard form.
        For the one level system we use the non-diagonal form because when performing the secular approximation one also has to keep coherences between different photon numbers when calculating the electronic dissipator
    """
    if isinstance(H, Operator):
        H = H.toarray()

    dim = H.shape[0]
    Id = np.eye(H.shape[0])

    # Writing the coherent evolution
    if method == "einsum":
        L = 1j * (
            np.einsum("ik,jl->ijkl", Id, H) - np.einsum("ik,jl->ijkl", H, Id)
        ).reshape((dim**2, dim**2))
    elif method == "kron":
        L = 1j * (np.kron(Id, H) - np.kron(H, Id))

    if c_ops is not None:
        L += dissipator(c_ops, method, diagonal_form)

    if cond:
        c = la.cond(L)
        print("Condition number of Liouvillian: ", c)

    return L
