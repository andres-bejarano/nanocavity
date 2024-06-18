import numpy as np
import nanocavity.distributions as ndist
import numpy.linalg as la
import copy

def matrix_elements(A, v, g):
    r""" Construction of an operator in given basis.

    Parameters
    ----------
    A: list of operators,
    v: numpy array with basis vectors.
    g: coupling to each level included in A

    Returns
    ----------
    M: numpy array with the information of each operator in A written in
    the basis of v.
    """
    if not isinstance(A, (list, tuple)):
        A = [A]
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
    if not isinstance(E, np.ndarray):
        E = np.array(E)

    if not isinstance(mu, np.ndarray):
        mu = np.array(mu)
    
    DE = E.reshape(1, -1, 1) - E.reshape(1, 1, -1)
    mu = mu.reshape(-1, 1, 1) 
    f = ndist.fermi_dirac(E=DE, kT=kT, mu=mu)
    return f

def bose_matrix(E, kT=0.1):
    if not isinstance(E, np.ndarray):
        E = np.array(E)
    DE = E.reshape(-1, 1) - E.reshape(1, -1)
    nb = ndist.bose_einstein(E=DE, kT=kT)
    np.fill_diagonal(nb, 0)
    return nb

def transition_rate(E, v, A, g, kT=0.1, mu=0, bath='fermionic'):
    """
    Calculates the transition rates between many-body states to be
    used in rate equtions. Returns a 3d numpy array with axis 0
    corresponding to the chemical potential and axis 1 and 2
    to the eigenstates.

    Parameters
    ----------
    E: system eigenvalues
    v: system eigenvectors
    A: list of all annihilation operators which interact with the
        considered bath
    g: coupling to the each level considered in A
    kT: temperature
    mu: chemical potential of the lead (only relevant for fermionic baths)
    bath: considering a fermionic or bosonic bath

    Returns
    ---------
    rates_p: transition rate matrix for adding particles to the central system
    rates_m: transition rate matrix for removing particles to
                the central system
    """
    if not isinstance(mu, np.ndarray):
        mu = np.array(mu)
    M = matrix_elements(A, v, g)

    np.fill_diagonal(M, 0)
    a, b = np.nonzero(M)  # Indices for the Bohr frequencies appearing in f^-
    mask = np.full((M.shape[0], M.shape[1]), False)
    mask[a, b] = True
    a2, b2 = np.nonzero(M.T)  # Inds. for the Bohr frequencies appearing in f^+

    DE = E.reshape(-1, 1) - E.reshape(1, -1)

    mu = mu.reshape(-1, 1)
    DElistp = DE[a2, b2].copy()
    DElistm = DE[a, b].copy()

    if bath == 'fermionic':
        f_mat_p = np.zeros((mu.shape[0], M.shape[0], M.shape[1]))
        f_mat_m = np.zeros((mu.shape[0], M.shape[0], M.shape[1]))
        f_list_p = ndist.fermi_dirac(DElistp, kT=kT, mu=mu)
        f_list_m = 1 - ndist.fermi_dirac(-DElistm, kT=kT, mu=mu)

        f_mat_p[:, mask.T] = f_list_p
        f_mat_m[:, mask] = f_list_m
        G_p = f_mat_p * M.T
        G_m = f_mat_m * M
        return G_p, G_m

    elif bath == 'bosonic':
        n_mat_p = np.zeros((M.shape[0], M.shape[1]))
        n_mat_m = np.zeros((M.shape[0], M.shape[1]))
        n_list_p = ndist.bose_einstein(DElistp, kT=kT)
        n_list_m = 1 + ndist.bose_einstein(-DElistm, kT=kT)

        n_mat_p[mask.T] = n_list_p
        n_mat_m[mask] = n_list_m
        K_p = n_mat_p * M.T
        K_m = n_mat_m * M
        return K_p, K_m

def bath_system_bath_rate(E, v, A, m, VL, VR, kT=0.1):
    if not isinstance(E, np.ndarray):
        E = np.array(E)
    if not isinstance(VL, np.ndarray):
        VL = np.array(VL)
    if not isinstance(VR, np.ndarray):
        VR = np.array(VR)

    M = matrix_elements(A, v, m)
    DE = abs(E.reshape(1, 1, -1, 1) - E.reshape(1, 1, 1, -1))
    V = abs(VL.reshape(-1, 1, 1, 1) - VR.reshape(1, -1, 1, 1))
    Fp = ndist.Fermi_cb(V-DE, kT) + ndist.Fermi_cb(-V-DE, kT)
    Fm = ndist.Fermi_cb(V+DE, kT) + ndist.Fermi_cb(-V+DE, kT)
    Mp = Fp * M.conj().T
    Mm = Fm * M
    return Mp, Mm


def populations(rates):
    r"""
    Computes the stationary solution of the rate equation \Gamma P = 0

    Parameters
    -----------
    rates: list of ndarrays
        A list containing one transition rate matrix per considered lead

    Returns
    ----------
    P: Populations
    """
    if not isinstance(rates, list):
        rate = rates.copy()
        rate = [rate]
    else:
        rate = copy.deepcopy(rates)

    for i in range(len(rate)):
        front = np.ones(len(rate) - i - 1)
        back = np.ones(len(rate) - 1 - front.shape[0])
        rate[i].shape = np.insert(rate[i].shape, 0, front)
        rate[i].shape = np.insert(rate[i].shape, -2, back)

    Gamma = sum(rate)
    k = Gamma.shape[-1]

    column_sum = Gamma.sum(axis=-2)
    for i in range(k):
        Gamma[..., i, i] = -column_sum[..., i]

    Gamma[..., k-1, :] = 1
    b_dims = Gamma.shape[0:-1]
    b = np.zeros(b_dims)
    b[..., k-1] = 1

    P = la.solve(Gamma, b)
    return P


def electro_current2(Gd, P, electrode=0):
    r"""
    Stationary current flowing through a given electrode

    Parameters:
    ------------
    Gd: nd-array
        Difference of transition rate matrices for
        adding and removing a particle
        Gd = G_p - G_m
    P: nd-array
        Populations; stationary solution of the rate equation
    electrode: int
        specifying the electrode

    Returns:
    -------
    I: nd_array
        current flowing through the specified electrode
    """
    axes = np.arange(P.ndim)
    axes[0], axes[electrode] = electrode, 0
    P = np.transpose(P, axes=axes)
    I = np.einsum('ijk,i...k->i...', Gd, P)
    I = np.transpose(I, axes=axes[:-1])
    return I


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


def photo_current2(Kd, P):
    """ Function calculating the emitted photons of a nanocavity

    Parameters
    --------
    Kd: nd-array
        Difference between the transition bosonic rate matrices for adding
        and substracting a particle to/ from the central system
    P: nd-array
        nd-aray containing the stationary populations

    Returns
    ---------
    I_ph: emitted photons
    """
    return np.einsum('ab, ...b->...', Kd, P)

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


def power_spectrum2(Kp, Km, P, E, omega):
    DE = E.reshape(-1, 1) - E.reshape(1, -1)
    omega = omega.reshape(-1, 1, 1)
    Lm = ndist.lorentzian(-DE - omega, w=Km)
    Lp = ndist.lorentzian(DE - omega, w=Kp)
    D = Km[None]*Lm - Kp[None]*Lp
    return np.einsum('iab,...b->i...', D, P)


def power_spectrum(Kp, Km, P, E, omega, state_resolved=False):
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
    # print(E.shape)
    DE = E.reshape(1, -1, 1) - E.reshape(1, 1, -1)
    # print(DE.shape)
    omega = omega.reshape(-1, 1, 1)
    Lm = ndist.lorentzian(-DE - omega, w=Km)
    Lp = ndist.lorentzian(DE - omega, w=Kp)
    Km = Km[np.newaxis]
    Kp = Kp[np.newaxis]
    D = Km * Lm - Kp * Lp
    # print(D.shape)
    # print(P.shape)
    #Ensuring that the diagonal is exactly zero
    for i in range(omega.shape[0]):
        np.fill_diagonal(D[i, : , :], 0)
    if state_resolved:
        return np.einsum('iab,jkb->bijk', D, P)
    return np.einsum('iab,jkb->ijk', D, P)
