import numpy as np
from scipy.linalg import eig

def eig_norm(L):
    El, vl, vr = eig(L.full(), left=True)
    norm = np.einsum("ai,ai->i", vl.conj(), vr) ** -0.5
    vl *= norm
    vr *= norm
    return El, vl, vr
