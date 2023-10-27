import numpy as np
import nanocavity.distributions as ndist
import numpy.linalg as la


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
    g = np.array(g).reshape(len(A), len(A))
    M = []
    for i in range(len(A)):
        M.append(A[i].inner(v))
    M = np.array(M)
    MGM = np.einsum('iab,ij,jab->ab', M.conj(), g, M)
    return MGM

def fermi_matrix(E, kT=0.1, mu=0):
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
    DE = E.reshape(1, -1, 1) - E.reshape(1, 1, -1)
    mu = mu.reshape(-1, 1, 1) 
    f = ndist.fermi_dirac(E=DE, kT=kT, mu=mu)
    return f

def bose_matrix(E, kT=0.1):
    E = np.array(E)
    DE = E.reshape(-1, 1) - E.reshape(1, -1)
    nb = ndist.bose_einstein(E=DE, kT=kT)
    np.fill_diagonal(nb, 0)
    return nb


def transition_rate(E, v, A, g, kT=0.1, mu=[0], bath='fermionic'):
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
    M = matrix_elements(A, v, g)
    if bath=='fermionic':
        fpr = fermi_matrix(E, kT=kT, mu=mu)
        fmr = 1 - fermi_matrix(-E, kT=kT, mu=mu)
        Gpr = fpr * M.conj().T
        Gmr = fmr * M
        return Gpr, Gmr
    elif bath=='bosonic':
        np = bose_matrix(E, kT=kT)
        nm = 1 + bose_matrix(-E, kT=kT)
        Kp = np * M.conj().T
        Km = nm * M
        return Kp, Km

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

def bath_system_bath_rate(E, v, A, M, VL, VR, kT=0.1):
    M_elements = matrix_elements(A, v, M)[np.newaxis, np.newaxis]
    DE = E.reshape(1, 1, -1, 1) - E.reshape(1, 1, 1, -1)
    V = VL.reshape(-1, 1, 1, 1) - VR.reshape(1, -1, 1, 1)
    Fp = Fermi_cb(V-DE, kT)
    Fm = Fermi_cb(-V-DE, kT=kT)
    F = Fp + Fm
    return M_elements * F


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

def electro_current(DGi, P, electrode='left'):
    r"""
    Electro-current calculated in the lead left
    Parameters
    ----------
    DGi: difference between transition rate matrix: transition in the system due to the injection of particles from the electrode i (Gpi) and transition in the system due to the extraction of particles from the system (Gpi). DGi = Gpi - Gpi
    P: stationray solution of rate equation, populations
    electrode: string specifying left or right electrode
    Return 
    ----------
    I: electro-current 
    """
    if electrode=='left':
        return np.einsum('iab,ijb->ij', DGi, P)
    elif electrode=='right':
        return np.einsum('jab,ijb->ij', DGi, P)

def photo_current(Kp, Km, P):
    r"""
    photo-current calculated 
    Parameters
    ----------
    Kp: nxn numpy array where n is the dimension of hilbert space. It Is the outcoming transition rate
    Km: nxn numpy array where n is the dimension of hilbert space. It is the incoming transition rate
    P: mxqxn numpy array where m,q is the dimension of the left,right hand side chemical potential and n the dimension of hilbert space. P is the population
    Return 
    ----------
    I: mxq numpy array.
    """
    return np.einsum('ab,ijb->ij', Km - Kp, P)

def power_spectrum(Kp, Km, P, E, omega):
    r""" power spectrum for system whose elements in the master equation corresponding to transition frequencies satisfy $\omega_{ab}-\omega_{cd}<< 1/tau{sys}$.(Secular approximation)

    Parameters
    ----------
    Kp: transition rate matrix for a transition in the system due to the injection of particles from the environment.
    Km: transition rate matrix for a transition in the system due to the extraction of particles from the system.
    P: stationray solution of rate equation, populations
    E: eigenstates of the system hamiltonian
    omega: frequency of the emitted light

    Return 
    ----------
    I: power spectrum map depending on left and right voltage $V_L$, $V_R$ and frequency of the emitted light $\omega$
    """
    DE = E.reshape(1, -1, 1) - E.reshape(1, 1, -1)
    omega = omega.reshape(-1, 1, 1)
    Lm = ndist.lorentzian(-DE - omega, w=Km)
    Lp = ndist.lorentzian(DE - omega, w=Kp)
    Km = Km[np.newaxis]
    Kp = Kp[np.newaxis]
    D = Km * Lm - Kp * Lp
    #Ensuring that the diagonal is exactly zero
    for i in range(omega.shape[0]):
        np.fill_diagonal(D[i, : , :], 0)
    return np.einsum('iab,jkb->ijk', D, P)
