import numpy as np
import nanocavity.operators as no
import nanocavity.rate_equation as nre
import nanocavity.master_equation as nme
import secondquant as sq


def Hamiltonian(Eg, Delta, hw_ph, g_ph, U=0, rwa=False, max_bosons=1, ret_nop=False):
    """
    Function calculating the Hamiltonian describing a TLS to a
    cavity.
    Utilizes secondquant operators

    Parameters:
    -------
    Eg: Float
        Ground state Energy
    Delta: Float
        Splitting between ground and excited state
    hw_ph: Float
        energy of the cavity (photon) mdoe
    g_ph: Float
        coupling strength to the cavity
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
    anni_list: list
        List containing the annihilation operators
    num_list: list
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
    anni_list = [dg, de, a]
    if ret_nop:
        num_list = [Nfg, Nfe, Nb]
        return H0, Hint, anni_list, num_list
    return H0, Hint, anni_list

def H_vi(Eg, Delta, hw_ph, g_ph, hw_vi, g_vi, U, max_bosons, rwa=False):
    """
    Function calculating the Hamiltonian describing a TLS to a
    cavity and a vibronic environment.
    Utilizes secondquant operators

    Parameters:
    -------
    Eg: Float
        Ground state Energy
    Delta: Float
        Splitting between ground and excited state
    hw_ph: Float
        energy of the cavity (photon) mdoe
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
    anni_list: list
        List containing the annihilation operators
    num_list: list
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
    anni_list = [dg, de, a_ph, a_vi]
    num_list = [ng, ne, n_ph, n_vi]
    return H, H0, Hint, anni_list, num_list

def collapses_vi(H, ops, VL, VR, kappa, Gamma_L, Gamma_R, kT):
    """
    Function to calculate the collapse operators with secondquant operators

    Parameters:
    -----
    H: secondquant operator
        Hamiltonian
    ops: list with 3 entries
        List containing the annihilation operators for the ground and
        excited state, as well as the phononic one
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

    Returns:
    -----
    c_ops: list
        List containing the collapse operators
    """
    dg = ops[0]
    de = ops[1]
    a_ph = ops[2]

    # left electrode
    c_gL = no.collapses(dg, H, kT, bath="fermionic", rate=Gamma_L, mu=VL)
    c_eL = no.collapses(de, H, kT, bath="fermionic", rate=Gamma_L, mu=VL)
    CL = c_gL + c_eL

    # left electrode
    c_gR = no.collapses(dg, H, kT, bath="fermionic", rate=Gamma_R, mu=VR)
    c_eR = no.collapses(de, H, kT, bath="fermionic", rate=Gamma_R, mu=VR)
    CR = c_gR + c_eR

    # cavity mode
    CA = no.collapses(a_ph, H, kT, bath="bosonic", rate=kappa)

    c_ops = CL + CR + CA
    return c_ops

def collapses(H_parameters, VL, VR, kappa, Gamma_L, Gamma_R, kT, alone=True, iva=False):
    """
    Function to calculate the collapse operators with secondquant operators

    Parameters:
    -----
    H_parameters: Tuple 
        System parameters
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
    alone: Logic
        If false, it returns two lists with collapses for particle aggregation and elimination processes  
    iva: Logic
        Write the dissipator to the base without interaction 

    Returns:
    -----
    c_ops: list
        List containing the collapse operators
    """

    H0, Hint, [dg, de, a] = Hamiltonian(*H_parameters)

    if iva:
        H = H0
    else:
        H = H0 + Hint

    # left electrode
    c_gL = no.collapses(dg, H, kT, "fermionic", Gamma_L, mu=VL)
    c_eL = no.collapses(de, H, kT, "fermionic", Gamma_L, mu=VL)
    CL = c_gL + c_eL

    # right electrode
    c_gR = no.collapses(dg, H, kT, "fermionic", Gamma_R, mu=VR)
    c_eR = no.collapses(de, H, kT, "fermionic", Gamma_R, mu=VR)
    CR = c_gR + c_eR

    # cavity mode
    CA = no.collapses(a, H, kT, "bosonic", kappa)

    c_ops = CL + CR + CA

    if alone:
        return c_ops
    return [dg, de, a], H0, Hint, c_ops


def rho_st(
    H_parameters,
    VL,
    VR,
    kappa,
    Gamma_L,
    Gamma_R,
    kT,
    iva=False,
    method='msolve'):
    """
    Function that calculates the density matrix in the steady state following the Born-Markov master equation.

    Parameters:
    -----
    H_parameters: Tuple 
        System parameters
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
    iva: Logic
        Write the dissipator to the base without interaction 
    method: str
        By default 'msolve 'it uses full master equation approach master_equation.py, if it is 'rsolve' then it will use rate_quation.py
    
    Returns:
    -----
    rho_st: 2D array
        Density matrix in the stationary regime
    """
    H0, Hint, [dg, de, a] = Hamiltonian(*H_parameters)
    H = H0 + Hint
    if method == "msolve":
        c_ops = collapses(H_parameters, VL, VR, kappa, Gamma_L, Gamma_R, kT, iva=iva)
        L = no.liouvillian(H, list(c_ops))
        rho = nme.stationary(L)
    elif method == 'rsolve':
        E, V = H.eigh()
        # transtion rates, populations and spectrum
        Kp, Km = nre.transition_rate(E, V, a, kappa, kT, bath="bosonic")
        K = Kp + Km
        GpL, GmL = nre.transition_rate(E, V, [dg, de], Gamma_L * np.eye(2), VL, kT)
        GpR, GmR = nre.transition_rate(E, V, [dg, de], Gamma_R * np.eye(2), VR, kT)
        GL = (GpL + GmL)[:, None]  # VL, VR
        GR = (GpR + GmR)[None, :]
        Gamma = K[np.newaxis, np.newaxis] + GL + GR
        rho = nre.populations(Gamma)
    return rho


def correlation(H_parameters, VL, VR, kappa, Gamma_L, Gamma_R, kT, tlist, iva=False):
    """
    Function to calculate the firs order correlation function of two operators. <A(t)B(0)>.
    The calculation is carried out by performing quantum regression theorem. 

    Parameters:
    -----
    H_parameters: Tuple 
        System parameters
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
    tlist: ndarray
        Discretization in time domain 
     iva: Logic
        Write the dissipator to the base without interaction 
    Returns:
    -----
    correlation: 2D array
        The correlation between two operators in time 
    """
     
    H0, Hint, [_, _, a] = Hamiltonian(*H_parameters)
    H = H0 + Hint
    
    c_ops = no.collapses_tls(H_parameters, VL, VR, kappa, Gamma_L, Gamma_R, kT, iva=iva)
    L = no.liouvillian(H, c_ops)
    return nme.correlation_AB(L, a.d, a, tlist)

def spectrum_vi(H, op_list, VL, VR, kappa, Gamma_L, Gamma_R, kT, wlist, Hint=0):
    """
    Function to calculate the firs order correlation function of two operators. <A(t)B(0)>.
    The calculation is carried out by performing quantum regression theorem. 

    Parameters:
    -----
    H_parameters: Tuple 
        System parameters
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
    tlist: ndarray
        Discretization in time domain 
     iva: Logic
        Write the dissipator to the base without interaction 
    Returns:
    -----
    correlation: 2D array
        The correlation between two operators in time 
    """

    c_ops = collapses_vi(H - Hint, op_list, VL, VR, kappa, Gamma_L, Gamma_R, kT)
    L = no.liouvillian(H, c_ops)
    I = kappa * nme.spectrum(L, op_list[2], wlist)
    return I
  
def spectrum(
    H_parameters,
    VL,
    VR,
    kappa,
    Gamma_L,
    Gamma_R,
    kT,
    wlist,
    iva=False,
    method='msolve',
    **kwargs):
    """
    This function calculates the emission spectrum, based on the application of the quantum regression theorem on the first-order correlation of
    the operators that couple to the photon bath. The Fourier transform is applied to this first-order correlation function, resulting in a sum 
    of Lorentzians, which is implemented in the code below.
    -----
    H_parameters: Tuple 
        System parameters
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
    wlist: ndarray
        The discretization of frequencies
    iva: Logic
        Write the dissipator to the base without interaction 
    method: str
        By default 'msolve 'it uses full master equation approach master_equation.py, if it is 'rsolve' then it will use rate_quation.py
    Returns:
    -----
    spectrum: 2D array
        Emission spectrum
    """
    H0, Hint, [dg, de, a] = Hamiltonian(*H_parameters)
    H = H0 + Hint
    if method == 'msolve':
        c_ops = collapses(H_parameters, VL, VR, kappa, Gamma_L, Gamma_R, kT, iva=iva)
        L = no.liouvillian(H, c_ops)
        I = nme.spectrum(L, a, wlist, **kwargs)
        try:
            return I * kappa
        except:
            # ret_dat=True flag given, return also Mk and Ek
            return I[0] * kappa, I[1], I[2]
    elif method == 'rsolve':
        E, V = H.eigh()
        # transtion rates, populations and spectrum
        Kp, Km = nre.transition_rate(E, V, a, kappa, kT, bath="bosonic")
        K = Kp + Km
        GpL, GmL = nre.transition_rate(E, V, [dg, de], Gamma_L * np.eye(2), VL, kT)
        GpR, GmR = nre.transition_rate(E, V, [dg, de], Gamma_R * np.eye(2), VR, kT)
        GL = (GpL + GmL)[:, None]  # VL, VR
        GR = (GpR + GmR)[None, :]
        P = nre.populations(K[np.newaxis, np.newaxis] + GL + GR)
        return nre.power_spectrum(Kp, Km, P, E, wlist, **kwargs)
      

def g2(H_parameters, VL, VR, kappa, Gamma_L, Gamma_R, kT, tlist, iva=False):
    """
    This function calculates the second order corrrlation function, 
    based on the application of the quantum regression theorem 
    -----
    H_parameters: Tuple 
        System parameters
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
    tlist: ndarray
        The discretization of time
    iva: Logic
        Write the dissipator in the base without interaction 
    
    Returns
    -----
    g2: 2D array
        second order correlation function in time
    """
    H0, Hint, [_, _, a] = Hamiltonian(*H_parameters)
    H = H0 + Hint

    c_ops = collapses(H_parameters, VL, VR, kappa, Gamma_L, Gamma_R, kT, iva=iva
        )
    L = no.liouvillian(H, c_ops)
    _, cm = no.collapses(a, H, kT, bath="bosonic", rate=kappa, total=False)
    J = no.jump(cm)
    return nme.g2(L, J, tlist)
