import numpy as np
import nanocavity.qutip.operators as qo
import qutip as qt


def Hamiltonian(Eg, Delta, hw_ph, g_ph, U=0, rwa=False, max_bosons=1):
    """
    Function calculating the Hamiltonian describing a TLS to a cavity
    and a vibronic environment.
    Utilizes qutip operators

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
        U: Float
            Coulomb repulsion
        max_bosons: list of 2 ints
            List specifying the maximum numbers of bosons considered in the
            cavity and the vibronic mode


    Returns:
        ------
        H: qutip operator
            Total Hamiltonian
        H0: qutip operator
            Hamiltonian w/o interaciton between TLS and photons/vibrons
        Hint: qutip operator
            Interaction Hamiltonian
        anni_list: list
            List containing the annihilation operators
    """

    N = max_bosons + 1
    dg = qt.tensor(qt.fdestroy(2, 0), qt.qeye(N))
    de = qt.tensor(qt.fdestroy(2, 1), qt.qeye(N))

    # sigmaz = [[1, 0], [0, -1]] is playing the role of permutation as
    # |n_g, n_e> = -|n_e, n_g>
    a = qt.tensor(qt.sigmaz(), qt.sigmaz(), qt.destroy(N))

    He = (
        Eg * dg.dag() * dg
        + (Eg + Delta) * de.dag() * de
        + U * dg.dag() * de.dag() * de * dg
    )
    Hp = hw_ph * a.dag() * a
    H0 = He + Hp
    if rwa:
        Hint = g_ph * (a.dag() * dg.dag() * de + a * de.dag() * dg)
    else:
        Hint = g_ph * (a + a.dag()) * (dg.dag() * de + de.dag() * dg)
    anni_list = [dg, de, a]
    return H0, Hint, anni_list


def H_vi(Eg, Delta, hw_ph, g_ph, hw_vi, g_vi, U, max_bosons, rwa=False):
    """
    Function calculating the Hamiltonian describing a TLS to a cavity
    and a vibronic environment.
    Utilizes qutip operators

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


    Returns:
        ------
        H: qutip operator
            Total Hamiltonian
        H0: qutip operator
            Hamiltonian w/o interaciton between TLS and photons/vibrons
        Hint: qutip operator
            Interaction Hamiltonian
        anni_list: list
            List containing the annihilation operators
    """

    N_ph = max_bosons[0] + 1
    N_vi = max_bosons[1] + 1

    dg = qt.tensor(qt.fdestroy(2, 0), qt.qeye(N_ph), qt.qeye(N_vi))
    de = qt.tensor(qt.fdestroy(2, 1), qt.qeye(N_ph), qt.qeye(N_vi))
    a_ph = qt.tensor(qt.sigmaz(), qt.sigmaz(), qt.destroy(N_ph), qt.qeye(N_vi))
    a_vi = qt.tensor(qt.sigmaz(), qt.sigmaz(), qt.qeye(N_ph), qt.destroy(N_vi))

    He = (
        Eg * dg.dag() * dg
        + (Eg + Delta) * de.dag() * de
        + U * dg.dag() * dg * de.dag() * de
    )
    Hph = hw_ph * a_ph.dag() * a_ph
    if rwa:
        H_m_ph = g_ph * (a_ph.dag() * dg.dag() * de + a_ph * de.dag() * dg)
    else:
        H_m_ph = g_ph * (a_ph.dag() + a_ph) * (dg.dag() * de + de.dag() * dg)
    Hvi = hw_vi * a_vi.dag() * a_vi
    H_m_vi = g_vi * (a_vi.dag() + a_vi) * de.dag() * de

    H0 = He + Hph + Hvi
    H_int = H_m_ph + H_m_vi
    H = H0 + H_int

    anni_list = [dg, de, a_ph, a_vi]
    return H, H0, H_int, anni_list

def collapses_vi(H, ops, VL, VR, kappa, Gamma_L, Gamma_R, kT):
    """
    Function to calculate the collapse operators with qutip operators

    Parameters:
        -----
        H: qutip operator
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

    c_gL = qo.collapses(dg, H, kT, bath="fermionic", rate=Gamma_L, mu=VL)
    c_eL = qo.collapses(de, H, kT, bath="fermionic", rate=Gamma_L, mu=VL)
    CL = c_gL + c_eL

    c_gR = qo.collapses(dg, H, kT, bath="fermionic", rate=Gamma_R, mu=VR)
    c_eR = qo.collapses(de, H, kT, bath="fermionic", rate=Gamma_R, mu=VR)
    CR = c_gR + c_eR

    CA = qo.collapses(a_ph, H, kT, bath="bosonic", rate=kappa)

    c_ops = CL + CR + CA
    return c_ops


def collapses(
    H_parameters, VL, VR, kappa, Gamma_L, Gamma_R, kT, m=0, lead2lead=False, alone=True, iva=False):
    
    """
    Function to calculate the collapse operators with qutip  operators

    Parameters:
        -----
        H: qutip operator
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
        m: float
            coupling between electron tunneling as cavity mode
        lead2lead: logic
            Dissipator for a electron tunnleing interacting with the cavity mode
        alone: Logic
            If false, it returns two lists with collapses for particle aggregation and elimination processes  
        iva: Logic
            If True write the dissipator in the basis without interaction. Defaults to False.
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
    c_gL = qo.collapses(dg, H, kT, "fermionic", Gamma_L, mu=VL)
    c_eL = qo.collapses(de, H, kT, "fermionic", Gamma_L, mu=VL)
    CL = c_gL + c_eL

    # right electrode
    c_gR = qo.collapses(dg, H, kT, "fermionic", Gamma_R, mu=VR)
    c_eR = qo.collapses(de, H, kT, "fermionic", Gamma_R, mu=VR)
    CR = c_gR + c_eR

    # cavity mode
    CA = qo.collapses(a, H, kT, "bosonic", kappa)

    if lead2lead:
        c_lead = qo.lead_cavity_lead_collapses(a, H, VL, VR, kT, m)
        c_ops = CL + CR + CA + c_lead
    else:
        c_ops = CL + CR + CA

    if alone:
        return c_ops
    return [dg, de, a], H0, Hint, c_ops

def correlation(H_parameters, VL, VR, kappa, Gamma_L, Gamma_R, kT, tlist, iva=False):
    """
    Function to calculate the firs order correlation function of two operators. <A(t)B(0)>.
    The calculation is carried out with the qutip functions based on quantum regression theorem

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
        If True write the dissipator in the basis without interaction. Defaults to False.    Returns:
    -----
    correlation: 2D array
        The correlation between two operators in time 
    """
    H0, Hint, [_, _, a] = Hamiltonian(*H_parameters)
    H = H0 + Hint

    c_ops = qo.collapses(H_parameters, VL, VR, kappa, Gamma_L, Gamma_R, kT, iva=iva)
    rho_st = qt.steadystate(H, list(c_ops))
    S = qt.correlation_2op_1t(
            H=H, state0=rho_st, taulist=tlist, c_ops=list(c_ops), a_op=a.dag(), b_op=a)
    return S

def spectrum_vi(H, op_list, VL, VR, kappa, Gamma_L, Gamma_R, kT, wlist, Hint=0):
    c_ops = collapses_vi(H, op_list, VL, VR, kappa, Gamma_L, Gamma_R, kT)
    a = op_list[2]
    I = kappa / (2 * np.pi) * qt.spectrum(H, wlist, list(c_ops), a.dag(), a)
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
):
    """
    Function to calculate the emission spectrum based in qutip scripts. The calculation is carried out 
    by performing the quantum regression theorem in the first order correlation function, then the numerical
    fourier transform is performed. 
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
        If True write the dissipator in the basis without interaction. Defaults to False.    Returns:
    -----
     spectrum: 2D array
        Emission spectrum
    """
    H0, Hint, [_, _, a] = Hamiltonian(*H_parameters)
    H = H0 + Hint
    c_ops = collapses(H_parameters, VL, VR, kappa, Gamma_L, Gamma_R, kT, iva=iva)
    I = kappa / (2 * np.pi) * qt.spectrum(H, wlist, list(c_ops), a.dag(), a)
    return I

def g2(H_parameters, VL, VR, kappa, Gamma_L, Gamma_R, kT, tlist, iva=False):
    """
    This function calculates the second order corrrlation function, 
    based on the application of the quantum regression theorem with qutip scripts.
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
        If True write the dissipator in the basis without interaction. Defaults to False.    
    Returns
    -----
    g2: tuple
        second order correlation function in time
    """
    H0, Hint, [_, _, a] = Hamiltonian(*H_parameters)
    H = H0 + Hint
    c_ops = collapses(H_parameters, VL, VR, kappa, Gamma_L, Gamma_R, kT, iva=iva)
    rho_st = qt.steadystate(H, list(c_ops))
    g2qt, _ = qt.coherence_function_g2(
    H, state0=rho_st, taulist=tlist, c_ops=c_ops, a_op=a, solver="me")
    return g2qt

