import numpy as np
from secondquant.composite import *
from secondquant.operator import Operator
from nanocavity.distributions import *
import numpy.linalg as la
import matplotlib.pyplot as plt


# E,v eigenvalues and eigenstates
# d operator which interacts with the  bath
# G tunneling rate between the bath and system
def transition_rate(E, v, d, G, mu):
    h = len(E)
    G = G.reshape(2, 2)
    DE = E.reshape(1, -1, 1) - E.reshape(1, 1, -1)
    mu = mu.reshape(-1, 1, 1)
    M = np.empty((len(d), h, h))
    
    for i in range(len(d)):
        M[i]  = d[i].inner(v)
    Mplus = np.einsum('iab,ij,jab->ab', M.conj(), G, M)
    Gamma_plus = fermi(DE, mu=mu) * \
            np.repeat(Mplus.conj().T.reshape(1, h, h), mu.shape[2], axis=0) 
    Gamma_minus = (1-fermi(-DE, mu=mu)) * \
            np.repeat(Mplus.reshape(1, h, h), mu.shape[2], axis=0)
    return Gamma_plus, Gamma_minus
