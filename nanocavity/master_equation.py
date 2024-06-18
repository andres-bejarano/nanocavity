import numpy as np
import nanocavity.distributions as ndist
import nanocavity.qutip.operators as qo
import nanocavity.operators as no
import nanocavity.rate_equation as nre
import qutip as qt
from scipy.linalg import eig
from secondquant.operator import Operator

def eig_norm(L):
    El, vl, vr = eig(L, left=True)
    norm = np.einsum("ai,ai->i", vl.conj(), vr) ** -0.5
    vl *= norm.conj()
    vr *= norm
    return El, vl, vr

def stationary(L):
    E, V = np.linalg.eig(L)
    # find the zero-eigenvalue mode index
    idx0 = np.argmin(np.abs(E))
    d = int(E.size ** .5)
    return V[:, idx0].reshape(d, d) / V[:, idx0].reshape(d, d).trace()

def current(J, L):
    J = J
    #w/v left/right eigenvectors
    El, vl, vr = eig_norm(L)
    index = np.argmin(np.abs(El))
    return np.dot(vl[:, index], np.dot(J, vr[:, index]))


def correlation_AB(L, A, B, tlist):
    if isinstance(A, Operator):
        A = A.toarray()

    if isinstance(B, Operator):
        B = B.toarray()

    dim = A.shape[0]
    Id = np.eye(dim)
    A = np.kron(A, Id)
    B = np.kron(B, Id)
    Rho_st = stationary(L).reshape(dim ** 2)
    S = np.zeros(len(tlist), dtype=np.complex128)

    El, vl, vr = eig_norm(L)

    w0 = np.eye(dim).reshape(1, dim ** 2)
    for k in range(0, len(El)):
        Ak = (np.dot(w0, A.dot(vr[:, k])))[0]
        Bk = np.dot(vl[:, k].conj(), B.dot(Rho_st))
        if abs(Ak) > 1e-12:
            S += Ak * Bk * np.exp(El[k] * tlist)
    return S

def spectrum(L, a, wlist, data=False):
    if isinstance(a, Operator):
        a = a.toarray()

    dim = a.shape[0]
    Id = np.eye(dim)
    Ad = np.kron(a.conj().T, Id)
    A = np.kron(a, Id)
    Rho_st = stationary(L).reshape(dim ** 2)
    I = np.zeros(len(wlist), dtype=np.complex128)

    El, vl, vr = eig_norm(L)

    w0 = np.eye(dim).reshape(1, dim ** 2)
    if data:
        print(f"{'k':4s} {'Ak.abs':12s} {'Bk.abs':12s} {'Mk.abs':12s} {'El[k].real':12s} {'El[k].imag':12s}")
    for k in range(0, len(El)):
        Ak = (np.dot(w0, Ad.dot(vr[:, k])))[0]
        Bk = np.dot(vl[:, k].conj(), A.dot(Rho_st))
        if abs(Ak) > 1e-12:
            if data:
                print(f"{k:4} {np.abs(Ak):12.6f} {np.abs(Bk):12.6f} {np.abs(Ak * Bk):12.6f} {El[k].real:12.6f} {El[k].imag:12.6f}")
            Dist = ndist.lorentzian(wlist - El[k].imag, - 2 * El[k].real)
            I += Ak * Bk * Dist
    return I.real

def g2(L, J, tlist, cutoff=1e-12):
    if isinstance(J, Operator):
        J = J.toarray()
    dim = J.shape[0]
    Rho_st = stationary(L).reshape(dim)
    G2 = np.zeros(len(tlist), dtype=np.complex128)
    El, vl, vr = eig_norm(L)
    w0 = np.eye(int(np.sqrt(dim))).reshape(1, dim)
    for k in range(0, len(El)):
        Ak = w0 @ J @ vr[:, k]
        Bk = vl[:, k].conj() @ J @ Rho_st
        if abs(Ak) > cutoff:
            G2 += Ak * Bk * np.exp(El[k] * tlist)
    G1 = w0 @ J @ Rho_st
    return G2 / G1 ** 2

