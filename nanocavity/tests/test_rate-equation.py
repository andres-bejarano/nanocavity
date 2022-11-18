import numpy as np
import numpy.linalg as la
from secondquant.composite import *
from secondquant.operator import Operator
from nanocavity.transition_rates import *
from nanocavity.distributions import *

def test_single_level():
    gL = 0.8
    gR = 0.2
    V = np.linspace(0, 1, 1)    
    #hamitlonian
    d, Nf = composite(fermion_modes=1)
    E, v = np.linalg.eigh(Nf[0].toarray())
    #rates
    GpL, GmL = transition_rate(E, v, d, gL , mu=V)
    GpR, GmR = transition_rate(E, v, d, gR, mu=V)
    GL, GR = transition_rate_matrix(GpL + GmL, GpR + GmR)

    #populations
    P = populations(GL + GR)
    
    #analytical result
    f = fermi(E=1.,  mu=V)
    P0 = (1 / (gL + gR)) * ((1 - f) * gL + (1 - f) * gR)
    P1 = (1. / (gL + gR)) * (f * gL + f * gR)
    PA = np.concatenate((P0, P1))
    assert np.allclose(P, PA)

    #electro-current
    IL = electro_current(GpL, GmL, P)
    IR = electro_current(GpR, GmR, P)
    
    #analytical reuslt
    fL = f.reshape(1, -1)
    fR = f.reshape(-1, 1)
    IA = ((gL * gR) /(gL + gR)) * (fL - fR)
    
    assert np.allclose(IL, IA)
    assert np.allclose(IR, IA)

def test_TLS():
    V = np.linspace(-12, 12, 101)
    N = len(V)

    #leads-exciton couplings
    gr = [[1,0], [0, 1]]

    #Two level system operators
    d, Nf = composite(fermion_modes=2)

    #Hamiltonian diagonalization
    delta = 1.
    H = (delta * Nf[1]).toarray()
    E, v = np.linalg.eigh(H)

    #transition rates matrix
    Gpr, Gmr = transition_rate(E, v, d, gr, mu=V)
    GL, GR = transition_rate_matrix(Gpr + Gmr, Gpr + Gmr)

    #populations
    P = populations(GL + GR)

    #current
    I = electro_current(Gpr, Gmr, P)
    
    #analytical

    pass

def test_conservation_populations():
    pass
