import numpy as np
import nanocavity.master_equation as nme
from scipy.linalg import eig

A = 0.4
B = 0.9
C = 1
D = 0.5
F = 3

def test_eig_norm():
    a = 1j * A - B
    b = 1j * C  - D
    c = 1j *  F 
    L = np.array([[a, c], [c,  b]])

    E, vl, vr = eig(L, left=True)

    E1, wr = np.linalg.eig(L)
    E2, wl = np.linalg.eig(L.conj().T)

    assert np.allclose(E, E1)
    assert np.allclose(E1, E2.conj())
    
    assert np.allclose(vl, wl)
    assert np.allclose(vr, wr)


    E, vl, vr = nme.eig_norm(L)

    norm = np.einsum("ai,ai->i", vl.conj(), vr) 
    
    assert np.allclose(norm.all(), 1)
