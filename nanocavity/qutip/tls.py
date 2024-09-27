import numpy as np
import nanocavity.qutip.operators as nqo
import qutip as qt


def Hamiltonian(Eg, Delta, hw_ph, g_ph, U=0, rwa=False, max_bosons=1):
    """
    Function calculating the Hamiltonian describing a TLS coupled to a cavity
    and a vibronic environment.
    Utilizes qutip operators

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
        rwa: logical
            Switch if rotating wave should be applied or not. Defaults to false
        max_bosons: int
            total number of photons

    Returns:
        ------
        H: qutip operator
            Total Hamiltonian
        H0: qutip operator
            Hamiltonian w/o interaciton between TLS and photons/vibrons
        Hint: qutip operator
            Interaction Hamiltonian
        anni_ops: list
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
    anni_ops = [dg, de, a]
    return H0, Hint, anni_ops


def H_vi(Eg, Delta, hw_ph, g_ph, hw_vi, g_vi, U, max_bosons, rwa=False):
    """
    Function calculating the Hamiltonian describing a TLS coupled to a cavity
    and a vibronic environment.
    Utilizes qutip operators

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


    Returns:
        ------
        H: qutip operator
            Total Hamiltonian
        H0: qutip operator
            Hamiltonian w/o interaciton between TLS and photons/vibrons
        Hint: qutip operator
            Interaction Hamiltonian
        anni_ops: list
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

    c_gL = nqo.collapses(dg, H, kT, bath="fermionic", rate=Gamma_L, mu=VL)
    c_eL = nqo.collapses(de, H, kT, bath="fermionic", rate=Gamma_L, mu=VL)
    CL = c_gL + c_eL

    c_gR = nqo.collapses(dg, H, kT, bath="fermionic", rate=Gamma_R, mu=VR)
    c_eR = nqo.collapses(de, H, kT, bath="fermionic", rate=Gamma_R, mu=VR)
    CR = c_gR + c_eR

    CA = nqo.collapses(a_ph, H, kT, bath="bosonic", rate=kappa)

    c_ops = CL + CR + CA
    return c_ops


def collapses(H, ops, VL, VR, kappa, Gamma_L, Gamma_R, kT, total=True):
    """
    Function to calculate the collapse operators with qutip  operators

    Parameters:
        -----
        H: qutip operator
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
        m: float
            coupling between electron tunneling as cavity mode
        lead2lead: logic
            Dissipator for a electron tunnleing interacting with the cavity mode
        total: Logic
            By default True, returns a list with all the collapses,
            if False it returns the collapses for creation (Plus) and elimination of particles (Minus).
    Returns:
        -----
        c_ops: List
            containing all collapses. If total True returns 2 lists. First all collapses for creation an excitation
        PLus: List
            In case total=True returns [c_gpL, c_epL, c_gpR, c_epR,  c_ap] list of collapses for creation of electrons/photons, each element is a list
        Minus: List
            In case total=True returns [c_gmL, c_emL, c_gmR, c_emR,  c_am] list of collapses for creation of electrons/photons, each element is a list
    """

    [dg, de, a] = ops

    # left electrode
    c_gpL, c_gmL = nqo.collapses(dg, H, kT, "fermionic", Gamma_L, mu=VL, total=False)
    c_epL, c_emL = nqo.collapses(de, H, kT, "fermionic", Gamma_L, mu=VL, total=False)

    # right electrode
    c_gpR, c_gmR = nqo.collapses(dg, H, kT, "fermionic", Gamma_R, mu=VR, total=False)
    c_epR, c_emR = nqo.collapses(de, H, kT, "fermionic", Gamma_R, mu=VR, total=False)

    # cavity mode
    c_ap, c_am = nqo.collapses(a, H, kT, "bosonic", kappa, total=False)

    if total:
        CL = c_gpL + c_epL + c_gmL + c_emL
        CR = c_gpR + c_epR + c_gmR + c_emR
        CA = c_ap + c_am
        return CL + CR + CA
    else:
        PLus = [c_gpL, c_epL, c_gpR, c_epR, c_ap]
        Minus = [c_gmL, c_emL, c_gmR, c_emR, c_am]
        return PLus, Minus
