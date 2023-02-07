import numpy as np
import numpy.linalg as la
from secondquant.composite import *
from secondquant.operator import Operator
from nanocavity.transition_rates import *
from nanocavity.distributions import *

def rate_parameters(modes=1):
    d, Nf = composite(fermion_modes=modes)
    if modes == 1: 
        g = 0.2
    else: 
        g = np.arange(modes ** 2).reshape(modes, modes)
    H = np.array(Nf).sum().toarray()
    e, v = la.eigh(H)
    return d, e, v, g

def test_matrix_elements():
    for i in range(1, 4):
        d, e, v, g =  rate_parameters(i)
        M = matrix_elements(d, v, g)
        assert np.allclose(M.shape,(len(v), len(v)))

def test_fermi_matrix():
    E = [0, 1, 2]
    mu = np.linspace(-1, 1, 3)
    A = fermi_matrix(E=E, mu=mu)  
    DE = np.array(E).reshape(1, -1, 1) - np.array(E).reshape(1, 1, -1)
    B = fermi_dirac(DE, mu=mu.reshape(-1, 1, 1))
    assert np.allclose(A, B)

def test_transition_rate():
    for i in range(1, 4):
        d, e, v, g = rate_parameters(i)
        Gp, Gm = transition_rate(e, v, d, g)
        assert np.allclose(Gp[0].diagonal(), 0)
        assert np.allclose(Gm[0].diagonal(), 0)
        
def test_populations():
    for i in range(1, 4):
        d, e, v, g = rate_parameters(i)
        Gp, Gm = transition_rate(e, v, d, g)
        GL, GR = transition_rate_matrix(Gp + Gm, Gp + Gm)
        P = populations(GL + GR)
        assert np.allclose(P[0, 0].sum(), 1)

def test_asymmetrical_bias():

    #chemical potential
    VL = np.linspace(-3, 3, 3)
    VR = np.linspace(-3, 2, 4)

    #leads-exciton couplings
    gl = [[1., 0], [0, 0.8]]
    gr = [[0.4, 0], [0, 0.2]]

    #Two level system operators
    d, Nf = composite(fermion_modes=2)

    #Hamiltonian diagonalization
    delta = 1.
    H = (delta * Nf[1]).toarray()
    E, v = np.linalg.eigh(H)

    #transition rates matrix
    GpL, GmL = transition_rate(E, v, d, gl, mu=VL) 
    GpR, GmR = transition_rate(E, v, d, gr, mu=VR)
    GL, GR = transition_rate_matrix(GpL + GmL, GpR + GmR)

    #populations
    Gamma = GL + GR
    P = populations(GL + GR)
    assert np.allclose(P.sum(axis=2), 1)

def test_electro_current_single_level():
    
    #chemical potential
    VL = np.linspace(-3, 3, 3)
    VR = np.linspace(-2, 4, 5)

    #leads-exciton couplings
    gl = 0.8
    gr = 0.2

    #Two level system operators
    d, Nf = composite(fermion_modes=1)

    #Hamiltonian diagonalization
    H = Nf[0].toarray()
    E, v = np.linalg.eigh(H)

    #transition rates matrix
    GpL, GmL = transition_rate(E, v, d, gl, mu=VL)
    GpR, GmR = transition_rate(E, v, d, gr, mu=VR)
    GL, GR = transition_rate_matrix(GpL + GmL, GpR + GmR)

    #populations
    P = populations(GL + GR)
    
    #current
    IL = electro_current(GpL - GmL, P, 'left')
    IR = electro_current(GpR - GmR, P, 'right')
    assert np.allclose(IL, -IR)

def test_electro_current_two_level():
    #chemical potential
    VL = np.linspace(-3, 2, 2)
    VR = np.linspace(-2, 1, 3)

    #leads-exciton couplings
    gl = [[1, 0], [0, 0.8]]
    gr = [[0.5, 0], [0, 0.3]]

    #Two level system operators
    d, Nf = composite(fermion_modes=2)

    #Hamiltonian diagonalization
    delta = 1.
    H = (delta * Nf[1]).toarray()
    E, v = np.linalg.eigh(H)

    #transition rates matrix
    GpL, GmL = transition_rate(E, v, d, gl, mu=VL)
    GpR, GmR = transition_rate(E, v, d, gr, mu=VR)
    GL, GR = transition_rate_matrix(GpL + GmL, GpR + GmR)

    #populations
    P = populations(GL + GR)

    #current
    IL = electro_current(GpL - GmL, P, 'left')
    IR = electro_current(GpR - GmR, P, 'right')
    
    assert np.allclose(IL, -IR)

