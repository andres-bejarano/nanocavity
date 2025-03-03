import numpy as np
import nanocavity.operators as no
import nanocavity.rate_equation as nre
import nanocavity.master_equation as nme
import nanocavity.distributions as ndist
import secondquant as sq


def Hamiltonian(Eg, Delta, hw_ph, g_ph, U, max_bosons, rwa=False, ret_nop=False):
    """
    Function calculating the Hamiltonian describing a TLS coupled to a cavity.
    Utilizes secondquant operators

    Parameters:
    -------
    Eg: Float
        Ground state Energy
    Delta: Float
        Splitting between ground and excited state
    hw_ph: Float
        energy of the cavity (photon) mode
    g_ph: Float
        coupling strength to the cavity
    U: Float
        Coulomb repulsion
    max_bosons: int
            total number of photons
    rwa: logical
        Switch if rotating wave should be applied or not. Defaults to false
    ret_nop: logical
            Switch if number operators should be returned. Defaults to False.

    Returns:
    ------
    H0: secondquant operator
        Hamiltonian w/o interaciton between TLS and photons/vibrons
    Hint: secondquant operator
        Interaction Hamiltonian
    anni_ops: list
        List containing the annihilation operators
    num_ops: list
        List containing the number operators
    """
    [dg, de, a], [Nfg, Nfe, Nb] = sq.composite(
        fermion_modes=2, boson_modes=1, max_bosons=max_bosons
    )
    He = Eg * Nfg + (Eg + Delta) * Nfe + U * dg.d * de.d * de * dg
    Hp = hw_ph * Nb
    H0 = He + Hp
    if rwa:
        Hint = g_ph * (a.d * dg.d * de + a * de.d * dg)
    else:
        Hint = g_ph * (a + a.d) * (dg.d * de + de.d * dg)
    anni_ops = [dg, de, a]
    if ret_nop:
        num_ops = [Nfg, Nfe, Nb]
        return H0, Hint, anni_ops, num_ops
    return H0, Hint, anni_ops


def H_vi(Eg, Delta, hw_ph, g_ph, hw_vi, g_vi, U, max_bosons, rwa=False):
    """
    Function calculating the Hamiltonian describing a TLS coupled to a
    cavity and a vibronic environment.
    Utilizes secondquant operators

    Parameters:
    -------
    Eg: Float
        Ground state Energy
    Delta: Float
        Splitting between ground and excited state
    hw_ph: Float
        energy of the cavity (photon) mode
    g_ph: Float
        coupling strength to the cavity
    hw_vi: Float
        Energy of the vibronic (phonon mode)
    g_vi: Float
        Coupling to the phonon mode
    U: Float
        Coulomb repulsion
    max_bosons: list of 2 ints
        List specifying the maximum numbers of bosons considered in the
        cavity and the vibronic mode
    rwa: logical
        Switch if rotating wave should be applied or not. Defaults to false

    Returns:
    ------
    H: secondquant operator
        Total Hamiltonian
    H0: secondquant operator
        Hamiltonian w/o interaciton between TLS and photons/vibrons
    Hint: secondquant operator
        Interaction Hamiltonian
    anni_ops: list
        List containing the annihilation operators
    num_ops: list
        List containing the number operators
    """
    [dg, de, a_ph, a_vi], [ng, ne, n_ph, n_vi] = sq.composite(
        fermion_modes=2, boson_modes=max_bosons
    )
    H_e = Eg * ng + (Eg + Delta) * ne + U * ng * ne
    H_ph = hw_ph * n_ph
    H_vi = hw_vi * n_vi
    if rwa:
        H_m_ph = g_ph * (a_ph.d * dg.d * de + a_ph * de.d * dg)
    else:
        H_m_ph = g_ph * (a_ph.d + a_ph) * (dg.d * de + de.d * dg)
    H_m_vi = g_vi * (a_vi.d + a_vi) * ne
    H0 = H_e + H_ph + H_vi
    Hint = H_m_vi + H_m_ph

    H = H0 + Hint
    anni_ops = [dg, de, a_ph, a_vi]
    num_ops = [ng, ne, n_ph, n_vi]
    return H, H0, Hint, anni_ops, num_ops


def collapses(H, ops, VL, VR, kappa, Gamma_L, Gamma_R, kT, hw_ph, total=True, cutoff=0):
    """
    Function to calculate the collapse operators with secondquant operators

    Parameters:
    -----
    H: secondquant operator
        Hamiltonian
    ops: list with 3 entries
        List containing the annihilation operators for the ground,
        excited and cavity modes
    VL: float
        bias at the left lead
    VR: float
        bias at the right lead
    kappa: float
        Cavity damping
    Gamma_L: float
        coupling of the left lead to the central system
    Gamma_R: float
        coupling of the right lead to the central system
    kT: float
        Temperature
    hw_ph: float
        energy of the cavity (photon) mode
    total: Logic
        By default True, returns a list with all the collapses,
        if False it returns the collapses for creation (Plus) and elimination of particles (Minus).

    Returns:
    -----
    c_ops: List
        containing all collapses. If total True returns 2 lists. First all collapses for creation an excitation
    Plus: List
        In case total=True returns [c_gpL, c_epL, c_gpR, c_epR,  c_ap] list of collapses for creation of electrons/photons, each element is a list
    Minus: List
        In case total=True returns [c_gmL, c_emL, c_gmR, c_emR,  c_am] list of collapses for creation of electrons/photons, each element is a list
    """

    [dg, de, a] = ops

    # we only diagonalize once, and keep the state ordering fixed
    basis = H.eigh()

    # left electrode
    c_gpL, c_gmL = no.collapses(dg, basis, kT, "fermionic", Gamma_L, mu=VL, total=False, cutoff=cutoff)
    c_epL, c_emL = no.collapses(de, basis, kT, "fermionic", Gamma_L, mu=VL, total=False, cutoff=cutoff)

    # right electrode
    c_gpR, c_gmR = no.collapses(dg, basis, kT, "fermionic", Gamma_R, mu=VR, total=False, cutoff=cutoff)
    c_epR, c_emR = no.collapses(de, basis, kT, "fermionic", Gamma_R, mu=VR, total=False, cutoff=cutoff)

    # cavity mode
    #c_ap, c_am = no.collapses(a, basis, kT, "bosonic", kappa, total=False, cutoff=cutoff)
    # we need to use the full cavity dissipator (see eq 6.37 in D.F.Walls and Gererd J. Milburn -  Quantum Optics).
    nb_p = ndist.bose_einstein(hw_ph, kT)
    nb_m = 1 + nb_p
    c_ap = [np.sqrt(kappa * nb_p) * a.d.toarray()]
    c_am = [np.sqrt(kappa * nb_m) * a.toarray()]

    if total:
        CL = c_gpL + c_epL + c_gmL + c_emL
        CR = c_gpR + c_epR + c_gmR + c_emR
        CA = c_ap + c_am
        return CL + CR + CA
    else:
        Plus = [c_gpL, c_epL, c_gpR, c_epR, c_ap]
        Minus = [c_gmL, c_emL, c_gmR, c_emR, c_am]
        return Plus, Minus


def rate_matrix(H, ops, VL, VR, kappa, Gamma_L, Gamma_R, kT, total=True):
    """
    Function that calculates the density matrix in the steady state following the Born-Markov master equation.
    Function to calculate the collapse operators with secondquant operators

    Parameters:
    -----
    H: secondquant operator
        Hamiltonian
    ops: list with 3 entries
        List containing the annihilation operators for the ground,
        excited and cavity modes
    VL: float
        bias at the left lead
    VR: float
        bias at the right lead
    kappa: float
        Cavity damping
    Gamma_L: float
        coupling of the left lead to the central system
    Gamma_R: float
        coupling of the right lead to the central system
    kT: float
        Temperature
    total: Logic
        By default True, returns the total transition rate matrix
        if False returns a list with each transition rate
    Returns:
    -----
        Gamma: ndarray
            Sum of individual transition rate matrices per lead
            Can either have a dimension for every lead or a fixed relation
            between biases in each lead.
            Shape: V_0 x V_1 x ... V_n x H.shape[0] x H.shape[1]
    """
    E, V = H.eigh()
    [dg, de, a] = ops

    # transtion rates, populations and spectrum
    Kp, Km = nre.transition_rate(E, V, a, kappa, kT, bath="bosonic")
    K = Kp + Km
    GpL, GmL = nre.transition_rate(E, V, [dg, de], Gamma_L * np.eye(2), VL, kT)
    GpR, GmR = nre.transition_rate(E, V, [dg, de], Gamma_R * np.eye(2), VR, kT)
    if total:
        K = Kp + Km
        GL = (GpL + GmL)[:, None]  # VL, VR
        GR = (GpR + GmR)[None, :]
        return K[np.newaxis, np.newaxis] + GL + GR
    else:
        Plus = [GpL, GpR, Kp]
        Minus = [GmL, GmR, Km]
        return Plus, Minus
