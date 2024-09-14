import numpy as np
import nanocavity.qutip.operators as qo
import nanocavity.operators as no
import nanocavity.rate_equation as nre
import nanocavity.master_equation as nme
import qutip as qt
import secondquant as sq

def Hnc(Eg, delta, omegac, coupling, u=0, rwa=True, max_bosons=1, ret_nop=False):
    [dg, de, a], [Nfg, Nfe, Nb] = \
        sq.composite(fermion_modes=2, boson_modes=1, max_bosons=max_bosons)
    He = Eg * Nfg + (Eg +  delta) * Nfe + u * dg.d * de.d * de * dg
    Hp = omegac * Nb
    H0 = He + Hp
    if rwa:
        Hint = coupling * (a.d * dg.d * de + a * de.d * dg)
    else:
        Hint = coupling * (a + a.d) * (dg.d * de + de.d * dg)
    H = H0 +  Hint
    L = [dg, de, a]
    if ret_nop:
        return H, L, [Nfg, Nfe, Nb]
    return H, L

def Hqt(Eg, delta, omegac, coupling, u=0, rwa=True, max_bosons=1):
    N = max_bosons + 1
    dg = qt.tensor(qt.fdestroy(2, 0), qt.qeye(N))
    de = qt.tensor(qt.fdestroy(2, 1), qt.qeye(N))

    #sigmaz = [[1, 0], [0, -1]] is playing the role of permutation as
    # |n_g, n_e> = -|n_e, n_g>
    a = qt.tensor(qt.sigmaz(), qt.sigmaz(), qt.destroy(N))

    He = Eg * dg.dag() * dg + (Eg + delta) * de.dag() * de +  u * dg.dag() * de.dag() * de * dg
    Hp = omegac * a.dag() * a
    H0 = He + Hp
    if rwa:
        Hint = coupling * (a.dag() * dg.dag() * de + a * de.dag() * dg)
    else:
        Hint = coupling * (a + a.dag()) * (dg.dag() * de + de.dag() * dg)
    H = H0 + Hint
    L = [dg, de, a]
    return H, L


def Hnc_vi(Eg, Delta, hw_ph, g_ph, hw_vi, g_vi, U, max_bosons, rwa=False):
    '''
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
    '''
    [dg, de, a_ph, a_vi], [ng, ne, n_ph, n_vi] = \
        sq.composite(fermion_modes=2, boson_modes=max_bosons)
    H_e = Eg * ng + (Eg+Delta) * ne + U*ng*ne
    H_ph = hw_ph*n_ph
    H_vi = hw_vi*n_vi
    if rwa:
        H_m_ph = g_ph * (a_ph.d*dg.d*de + a_ph*de.d*dg)
    else:
        H_m_ph = g_ph * (a_ph.d+a_ph) * (dg.d*de + de.d*dg)
    H_m_vi = g_vi * (a_vi.d + a_vi) * ne
    H0 = H_e + H_ph + H_vi
    Hint = H_m_vi + H_m_ph

    H = H0 + Hint
    anni_list = [dg, de, a_ph, a_vi]
    num_list = [ng, ne, n_ph, n_vi]
    return H, H0, Hint, anni_list, num_list


def Hqt_vi(Eg, Delta, hw_ph, g_ph, hw_vi, g_vi, U, max_bosons, rwa=False):
    '''
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
    '''

    N_ph = max_bosons[0] + 1
    N_vi = max_bosons[1] + 1

    dg = qt.tensor(qt.fdestroy(2, 0), qt.qeye(N_ph), qt.qeye(N_vi))
    de = qt.tensor(qt.fdestroy(2, 1), qt.qeye(N_ph), qt.qeye(N_vi))
    a_ph = qt.tensor(qt.sigmaz(), qt.sigmaz(), qt.destroy(N_ph), qt.qeye(N_vi))
    a_vi = qt.tensor(qt.sigmaz(), qt.sigmaz(), qt.qeye(N_ph), qt.destroy(N_vi))

    He = Eg*dg.dag()*dg + (Eg+Delta) * de.dag()*de +\
        U * dg.dag()*dg * de.dag()*de
    Hph = hw_ph * a_ph.dag() * a_ph
    if rwa:
        H_m_ph = g_ph * (a_ph.dag() * dg.dag() * de + a_ph * de.dag() * dg)
    else:
        H_m_ph = g_ph * (a_ph.dag()+a_ph) * (dg.dag()*de + de.dag()*dg)
    Hvi = hw_vi * a_vi.dag() * a_vi
    H_m_vi = g_vi * (a_vi.dag() + a_vi) * de.dag()*de

    H0 = He + Hph + Hvi
    H_int = H_m_ph + H_m_vi
    H = H0 + H_int

    anni_list = [dg, de, a_ph, a_vi]
    return H, H0, H_int, anni_list


def Hamiltonian(package, Eg, delta, omegac, coupling, u=0, rwa=True, max_bosons=1, ret_nop=False):
    if package=='nanocavity':
        return Hnc(Eg, delta, omegac, coupling, u, rwa, max_bosons, ret_nop)
    elif package=='qutip':
        return Hqt(Eg, delta, omegac, coupling, u, rwa, max_bosons)

def collapses_nc_vi(H, ops, VL, VR, kappa, gL, gR, kT):
    ''' 
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
        gL: float
            coupling of the left lead to the central system
        gR: float
            coupling of the right lead to the central system
        kT: float
            Temperature

        Returns:
        -----
        c_ops: list
            List containing the collapse operators
    '''
    dg = ops[0] 
    de = ops[1]
    a_ph = ops[2]

    #left electrode
    c_gL = no.collapses(dg, H, kT, bath='fermionic', rate=gL, mu=VL)
    c_eL = no.collapses(de, H, kT, bath='fermionic', rate=gL, mu=VL)
    CL = c_gL + c_eL

    #left electrode
    c_gR = no.collapses(dg, H, kT, bath='fermionic', rate=gR, mu=VR)
    c_eR = no.collapses(de, H, kT, bath='fermionic', rate=gR, mu=VR)
    CR = c_gR + c_eR

    #cavity mode
    CA = no.collapses(a_ph, H, kT, bath='bosonic', rate=kappa)

    c_ops = np.concatenate((CL, CR, CA))
    return c_ops

def collapses_qt_vi(H, ops, VL, VR, kappa, gL, gR, kT):
    ''' 
    Function to calculate the collapse operators with secondquant operators

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
        gL: float
            coupling of the left lead to the central system
        gR: float
            coupling of the right lead to the central system
        kT: float
            Temperature

    Returns:
        -----
        c_ops: list
            List containing the collapse operators
    '''
    dg = ops[0]
    de = ops[1]
    a_ph = ops[2]

    c_gL = qo.collapses(dg, H, kT, bath='fermionic', rate=gL, mu=VL)
    c_eL = qo.collapses(de, H, kT, bath='fermionic', rate=gL, mu=VL)
    CL = c_gL + c_eL

    c_gR = qo.collapses(dg, H, kT, bath='fermionic', rate=gR, mu=VR)
    c_eR = qo.collapses(de, H, kT, bath='fermionic', rate=gR, mu=VR)
    CR = c_gR + c_eR

    CA = qo.collapses(a_ph, H, kT, bath='bosonic', rate=kappa)

    c_ops = np.concatenate((CL, CR, CA))
    return c_ops

def collapses_nc(H_parameters, VL, VR, kappa, gL, gR, kT, alone=True, iva=False):

    H, [dg, de, a] = Hamiltonian('nanocavity', *H_parameters)

    if iva:
        coupling = H_parameters[3]
        Hint = coupling * (a.d * dg.d * de + a * de.d * dg)
        H -= Hint
    #left electrode
    c_gL = no.collapses(dg, H, kT, 'fermionic', gL, mu=VL)
    c_eL = no.collapses(de, H, kT, 'fermionic', gL, mu=VL)
    CL = c_gL + c_eL

    #right electrode
    c_gR = no.collapses(dg, H, kT, 'fermionic', gR, mu=VR)
    c_eR = no.collapses(de, H, kT, 'fermionic', gR, mu=VR)
    CR = c_gR + c_eR

    #cavity mode
    CA = no.collapses(a, H, kT, 'bosonic', kappa)

    c_ops = CL + CR + CA

    if alone:
        return c_ops
    if iva:
        return [dg, de, a], H + Hint, c_ops
    return [dg, de, a], H, c_ops


def collapses_qt(H_parameters, VL, VR, kappa, gL, gR, kT, m=0, lead2lead=False, alone=True, iva=False):

    H, [dg, de, a] = Hamiltonian('qutip', *H_parameters)

    if iva:
        coupling = H_parameters[3]
        Hint = coupling * (a.dag() * dg.dag() * de + a * de.dag() * dg)
        H -= Hint
    #left electrode
    c_gL = qo.collapses(dg, H, kT, 'fermionic', gL, mu=VL)
    c_eL = qo.collapses(de, H, kT, 'fermionic', gL, mu=VL)
    CL = c_gL + c_eL

    #right electrode
    c_gR = qo.collapses(dg, H, kT, 'fermionic', gR, mu=VR)
    c_eR = qo.collapses(de, H, kT, 'fermionic', gR, mu=VR)
    CR = c_gR + c_eR

    #cavity mode
    CA = qo.collapses(a, H, kT, 'bosonic', kappa)

    if lead2lead:
        c_lead = qo.lead_cavity_lead_collapses(a, H, VL, VR, kT, m)
        c_ops = CL + CR + CA + c_lead
    else:
        c_ops = CL + CR + CA

    if alone:
        return c_ops
    if iva:
        return [dg, de, a], H + Hint, c_ops
    return [dg, de, a], H, c_ops

def collapses(package, H_parameters, VL, VR, kappa, gL, gR, kT, m=0, lead2lead=False, alone=True, iva=False):
    if package=='nanocavity':
        return collapses_nc(H_parameters, VL, VR, kappa, gL, gR, kT, alone, iva)
    elif package=='qutip':
        return collapses_qt(H_parameters, VL, VR, kappa, gL, gR, kT, m=0, lead2lead=lead2lead, alone=alone, iva=iva)

def rho_st(package, H_parameters, VL, VR, kappa, gL, gR, kT, m=0, lead2lead=False, iva=False, full=True):
    H, [dg, de, a] = Hamiltonian(package, *H_parameters)
    if package=='nanocavity':
        if full:
            c_ops = collapses_nc(H_parameters, VL, VR, kappa, gL, gR, kT, iva=iva)
            L = no.liouvillian(H, list(c_ops))
            rho = nme.stationary(L)
        else:
            E, V = H.eigh()
            #transtion rates, populations and spectrum
            Kp, Km = nre.transition_rate(E, V,  a, kappa, kT, bath='bosonic')
            K = Kp + Km
            GpL, GmL = nre.transition_rate(E, V, [dg, de], gL*np.eye(2), VL, kT)
            GpR, GmR = nre.transition_rate(E, V, [dg, de], gR*np.eye(2), VR, kT)
            GL = (GpL + GmL)[:, None]  # VL, VR
            GR = (GpR + GmR)[None, :]
            Gamma = K[np.newaxis, np.newaxis] + GL + GR
            rho = nre.populations(Gamma)
    elif package=='qutip':
        c_ops = collapses_qt(H_parameters, VL, VR, kappa, gL, gR, kT, m, lead2lead, alone=True, iva=iva)
        rho = qt.steadystate(H, list(c_ops)).full() 
    return rho

def correlation(package, H_parameters, VL, VR, kappa, gL, gR, kT, tlist, iva=False):
    H, [_, _, a] = Hamiltonian(package, *H_parameters)
    if package=='nanocavity':
        c_ops = no.collapses_tls(H_parameters, VL, VR, kappa, gL, gR, kT, iva=iva)
        L = no.liouvillian(H, list(c_ops))
        S = nme.correlation_AB(L, a.d, a, tlist)
    
    elif package=='qutip':
        c_ops = qo.collapses_tls(H_parameters, VL, VR, kappa, gL, gR, kT, iva=iva)
        rho_st= qt.steadystate(H, list(c_ops))
        S = qt.correlation_2op_1t(H=H, state0=rho_st, taulist=tlist, c_ops=list(c_ops), a_op=a.dag(), b_op=a)
    return S

def spectrum_vi(package, H, op_list, VL, VR, kappa, gL, gR, kT, wlist, Hint=0):
    if package == 'nanocavity' or package == 'nc':
        c_ops = collapses_nc_vi(H-Hint, op_list, VL, VR, kappa, gL, gR, kT)
        L = no.liouvillian(H, c_ops)
        I = kappa * nme.spectrum(L, op_list[2], wlist)
    elif package == 'qutip' or package == 'qt':
        c_ops = collapses_qt_vi(H, op_list, VL, VR, kappa, gL, gR, kT)
        a = op_list[2]
        I = kappa / (2 * np.pi) * qt.spectrum(H, wlist, list(c_ops), a.dag(), a)
    else:
        print("Either calculate the spectrum with nanocavity or qutip")
        return 0
    return I

def spectrum(package, H_parameters, VL, VR, kappa, gL, gR, kT, wlist, iva=False, data=False, full=True):
    H, [dg, de, a] = Hamiltonian(package, *H_parameters)
    
    if package=='nanocavity':
        if full:
            c_ops = collapses('nanocavity', H_parameters, VL, VR, kappa, gL, gR, kT, iva=iva)
            L = no.liouvillian(H, list(c_ops))
            I = kappa * nme.spectrum(L, a, wlist, data=data)
        else:
            E, V = H.eigh()
            #transtion rates, populations and spectrum
            Kp, Km = nre.transition_rate(E, V,  a, kappa, kT, bath='bosonic')
            K = Kp + Km
            GpL, GmL = nre.transition_rate(E, V, [dg, de], gL*np.eye(2), VL, kT)
            GpR, GmR = nre.transition_rate(E, V, [dg, de], gR*np.eye(2), VR, kT)
            GL = (GpL + GmL)[:, None]  # VL, VR
            GR = (GpR + GmR)[None, :]
            P = nre.populations(K[np.newaxis, np.newaxis] + GL + GR)
            I = nre.power_spectrum(Kp, Km, P, E, wlist)
    if package=='qutip':
        c_ops = collapses('qutip', H_parameters, VL, VR, kappa, gL, gR, kT, iva=iva)
        I = kappa / (2 * np.pi) * qt.spectrum(H, wlist, list(c_ops), a.dag(), a)
    return I

def g2(package, H_parameters, VL, VR, kappa, gL, gR, kT, tlist, iva=False):
    H, [dg, de, a] = Hamiltonian(package, *H_parameters)
    if package=='nanocavity':
        c_ops = collapses('nanocavity', H_parameters, VL, VR, kappa, gL, gR, kT, iva=iva)
        L = no.liouvillian(H, list(c_ops))
        _, cm = no.collapses(a, H, kT, bath='bosonic', rate=kappa, total=False)
        J = no.jump(cm)
        return nme.g2(L, J, tlist)

    elif package=='qutip':
        c_ops = collapses('qutip', H_parameters, VL, VR, kappa, gL, gR, kT, iva=iva)
        rho_st= qt.steadystate(H, list(c_ops))
        g2qt, _ = qt.coherence_function_g2(H, state0=rho_st, taulist=tlist, c_ops=c_ops, a_op=a, solver='me')
        return g2qt

