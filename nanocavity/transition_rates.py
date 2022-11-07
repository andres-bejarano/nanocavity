import numpy as np
from secondquant.composite import *
from secondquant.operator import Operator
from nanocavity.distributions import *
import numpy.linalg as la
import matplotlib.pyplot as plt

def operator_basis(A, v):
    r""" Construction of an operator in given basis.
    
    Input variables:

        - [A]: list of operators,
        - [v]: numpy array with basis vectors.

    Output variables:
    
        - [M]: numpy array with the information of each operator in A written in the basis of v.
    """
    v = np.array(v)
    M = np.empty((len(A), len(v), len(v)))
    for i in range(len(A)):
          M[i]  = A[i].inner(v)
    return M

def fermi_matrix(E, mu=0):
    r""" Construction of numpy array whose matrix elements are fermi functions evaluated for each energy differences and chemical potential.
    
    Input variables:

        - [E]: all possible energy values,
        - [mu]: all possible chemical potential energies.
        
    Output variables:
    
        - [fermi]: fermi function evaluated for all combination of input variables.
    """
    E = np.array(E)
    mu = np.array(mu)
    N = len(mu)
    k = len(E)
    DE = E.reshape(1, -1, 1) - E.reshape(1, 1, -1)
    mu = mu.reshape(-1, 1, 1) 
    fL = fermi(E=DE, mu=mu).reshape(1, N, k, k)
    fR = fermi(E=DE, mu=mu).reshape(N, 1, k, k)
    return fL+fR

def transition_rate(E, v, A, g, mu=0):
    r""" trasition_rate construct a matrix numpy array with all possible transition rates, where each matrix element represent the transition rate  between two states at given chemical potential.

    Input variables

        - [E]: system eigenvalues,
        - [v]: system eigenvectors,
        - [A]: list of all operators which interacts with the environment,
        - [g]: list of all coupling values between each level and the environment, 
        - [mu]: all possible chemical potential values.
        
    Output variables:
    
        - [Gp]: transition rate matrix for a transition in the system due to the injection of particles from the environment.
        - [Gm]: transition rate matrix for a transition in the system due to the extraction of particles from the system.
    """
    g = np.array(g).reshape(2, 2)
    fp = fermi_matrix(E, mu=mu)
    fm = 1 - fermi_matrix(-E, mu=mu)
    M = operator_basis(A, v)
    G = np.einsum('iab,ij,jab->ab', M.conj(), g, M)
    Gp = fp * G.conj().T
    Gm = fm * G
    return Gp, Gm
