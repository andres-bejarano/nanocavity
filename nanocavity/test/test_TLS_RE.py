import numpy as np
import numpy.linalg as la
from secondquant.composite import *
from secondquant.operator import Operator
from nanocavity.transition_rates import *

def test_TLS():
    
    #chemical potential
    V = np.linspace(-12, 12, 101)
    
    #leads-exciton couplings
    gr = [[1,0], [0, 1]]
    
    #Two level system operators
    d, Nf = composite(fermion_modes=2)
    
    #Hamiltonian diagonalization
    E, v = np.linalg.eigh(Nf[1].toarray())
    
    #transition rates matrix
    Gpr, Gmr = transition_rate(E, v, d, gr, mu=V)
    GL, GR = transition_rate_matrix(Gpr + Gmr, Gpr + Gmr)
    
    #populations
    P = populations(GL + GR)
    
    #current
    I = electro_current(Gpr, Gmr, P)

    #saving data
    dataTLN = np.empty((len(V), len(V) + 1))
    dataTLN[:, 0] = V
    dataTLN[:, 1:] = I
    
    #read analytical solutions data
    dataTLA = np.loadtxt("TLA.anlytical.txt", delimiter=" ")
    ITLA = dataTLA[:, 1:]
    ITLN = dataTLN[:, 1:]
    assert np.allclose(ITLA, ITLN)
