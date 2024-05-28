import numpy as np
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
