import numpy as np
from secondquant.composite import *
from secondquant.operator import Operator
from nanocavity.distributions import *
import numpy.linalg as la
import matplotlib.pyplot as plt

def matrix_elements(A, v, g):
    r""" Construction of an operator in given basis.
    Parameters
    ----------
    A: list of operators,
    v: numpy array with basis vectors.
    
    Returns
    ----------
    M: numpy array with the information of each operator in A written in the basis of v.
    """
    v = np.array(v)
    g = np.array(g).reshape(len(A), len(A))
    M = np.empty((len(A), len(v), len(v)))
    for i in range(len(A)):
          M[i]  = A[i].inner(v)
    MGM = np.einsum('iab,ij,jab->ab', M.conj(), g, M)
    return MGM

def fermi_matrix(E, kBT=0.1, mu=0):
    r""" Construction of numpy array whose matrix elements are fermi functions evaluated for each energy differences and chemical potential.
    Parameters
    ----------
    E: all possible energy values,
    mu: all possible chemical potential energies.

    Returns
    ----------
    fermi: fermi function evaluated for all combination of input variables.
    """
    E = np.array(E)
    mu = np.array(mu)
    N = len(mu)
    k = len(E)
    DE = E.reshape(1, -1, 1) - E.reshape(1, 1, -1)
    mu = mu.reshape(-1, 1, 1) 
    f = fermi(E=DE, kBT=kBT, mu=mu)
    return f

def transition_rate(E, v, A, g, kBT=0.1, mu=[0]):
    r""" trasition_rate construct a matrix numpy array with all possible transition rates, where each matrix element represent the transition rate  between two states at given chemical potential.
    Parameters
    ----------
    E: system eigenvalues,
    v: system eigenvectors,
    A: list of all operators which interacts with the environment,
    g: list of all coupling values between each level and the environment, 
    mu: all possible chemical potential values.
        
    Returns
    ----------
    Gpr: transition rate matrix for a transition in the system due to the injection of particles from the environment.
    Gmr: transition rate matrix for a transition in the system due to the extraction of particles from the system.
    """
    fpr = fermi_matrix(E, kBT=kBT, mu=mu)
    fmr = 1 - fermi_matrix(-E, kBT=kBT, mu=mu)
    M = matrix_elements(A, v, g)
    Gpr = fpr * M.conj().T
    Gmr = fmr * M
    return Gpr, Gmr

def transition_rate_matrix(GL, GR):
    r""""The sum of transition rates corresponding to two environments must contain all possible values of chemical potential. The case of two left/right environments has been implemented so far.
    Parameters
    ----------
    GL: coming from the output of transtion_rate for left environment
    GR: coming from the output of transtion_rate for right environment

    Return
    ----------
    In progress
    """
    #vi: number of chemical potential values for lead i
    #n: system hamiltonian dimension

    vl, k, _ = GL.shape
    vr = GR.shape[0]

    GL = GL.reshape(vl, 1, k, k)
    GR = GR.reshape(1, vr, k, k)
    return GL, GR

def populations(Gamma):
    r""" The stationary solution of rate equation will calculated \Gamma P = 0.

    Parameters
    ----------
    Gamma: Transition rates matrix which contain all possible environments

    Return 
    ----------
    P: populations
    """
    vl, vr, k, _ = Gamma.shape

    #The diagonal of transition rate matrix is the - the sum of each column per each bias voltage vl, vr
    for i in range(k):
        Gamma[:, :, i, i] = -Gamma.sum(axis=2)[:, :, i]
    
    
    #conservation of probability \sum_iP_i=1 implies that one equation must be equal to 1
    Gamma[:, : , k - 1, :] = 1 
    b = np.zeros((vl, vr, k))
    b[:, :, k - 1] = 1
    
    P = la.solve(Gamma, b)
    return P

def electro_current(Gpr, Gmr, P):
    r"""
    Electro-current calculated in the lead r = left or right 
    Parameters
    ----------
    Gpr: transition rate matrix for a transition in the system due to the injection of particles from the environment. 
    Gmr: transition rate matrix for a transition in the system due to the extraction of particles from the system. 
    See nanocavity.transition_rates.transition_rates


    Return 
    ----------
    I: electro-current 
    """
    I = np.einsum('iab,ijb->ij', Gpr - Gmr, P)
    return I
