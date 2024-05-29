import numpy as np
import nanocavity.distributions as ndist
from scipy.linalg import eig

def eig_norm(L):
    El, vl, vr = eig(L, left=True)
    norm = np.einsum("ai,ai->i", vl.conj(), vr) ** -0.5
    vl *= norm
    vr *= norm
    return El, vl, vr

def stationary(L):
    #E, V = eig(L)
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

def spectrum(L, a, wlist):
    dim = a.shape[0]
    Id = np.eye(dim)
    Ad = np.kron(a.d.toarray(), Id)
    A = np.kron(a.toarray(), Id)
    Rho_st = stationary(L).reshape(64)
    I = np.zeros(len(wlist), dtype=np.complex128)

    El, vl, vr = eig(L, left=True)
    w0 = np.eye(8).reshape(1, 64)
    print(f"{'k':4s} {'Ak.abs':12s} {'Bk.abs':12s} {'Mk.abs':12s} {'El[k].real':12s} {'El[k].imag':12s}")
    for k in range(0, len(El)):
        Ak = (np.dot(w0, Ad.dot(vr[:, k])))[0]
        Bk = np.dot(vl[:, k], A.dot(Rho_st))
        if abs(Ak) > 1e-12:
            print(f"{k:4} {np.abs(Ak):12.6f} {np.abs(Bk):12.6f} {np.abs(Ak * Bk):12.6f} {El[k].real:12.6f} {El[k].imag:12.6f}")
            Dist = ndist.lorentzian(wlist + np.imag(El[k]), - 2 * np.real(El[k]))
            I += np.real(Ak * Bk * Dist)
    return I.real
