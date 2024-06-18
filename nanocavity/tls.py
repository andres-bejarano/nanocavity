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


def Hamiltonian(package, Eg, delta, omegac, coupling, u=0, rwa=True, max_bosons=1, ret_nop=False):
    if package=='nanocavity':
        return Hnc(Eg, delta, omegac, coupling, u, rwa, max_bosons, ret_nop)
    elif package=='qutip':
        return Hqt(Eg, delta, omegac, coupling, u, rwa, max_bosons)


def collapses_nc(H_parameters, VL, VR, kappa, gL, gR, kT, alone=True, iva=False):

    H, [dg, de, a] = Hamiltonian('nanocavity', *H_parameters)

    if iva:
        coupling = H_parameters[3]
        Hint = coupling * (a.d * dg.d * de + a * de.d * dg)
        H -= Hint
    #left electrode
    c_gL = no.collapses(dg, H, kT, bath='fermionic', mu=VL)
    c_eL = no.collapses(de, H, kT, bath='fermionic', mu=VL)
    CL = np.sqrt(gL) * np.array(c_gL + c_eL)

    #right electrode
    c_gR = no.collapses(dg, H, kT, bath='fermionic', mu=VR)
    c_eR = no.collapses(de, H, kT, bath='fermionic', mu=VR)
    CR = np.sqrt(gR) * np.array(c_gR + c_eR)

    #cavity mode
    CA = no.collapses(a, H, kT, bath='bosonic')

    CA = np.sqrt(kappa) * np.array(CA)

    c_ops = np.concatenate((CL, CR, CA))

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
    c_gL = qo.collapses(dg, H, kT, bath='fermionic', mu=VL)
    c_eL = qo.collapses(de, H, kT, bath='fermionic', mu=VL)
    CL = np.sqrt(gL) * np.array(c_gL + c_eL)

    #right electrode
    c_gR = qo.collapses(dg, H, kT, bath='fermionic', mu=VR)
    c_eR = qo.collapses(de, H, kT, bath='fermionic', mu=VR)
    CR = np.sqrt(gR) * np.array(c_gR + c_eR)

    #cavity mode
    CA = qo.collapses(a, H, kT, bath='bosonic')

    CA = np.sqrt(kappa) * np.array(CA)

    if lead2lead:
        c_lead = qo.lead_cavity_lead_collapses(a, H, VL, VR, kT, m)
        c_ops = np.concatenate((CL, CR, CA, c_lead))
    else:
        c_ops = np.concatenate((CL, CR, CA))

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



def correlation(package, H_parameters, VL, VR, kappa, gL, gR, kT, tlist, iva=False):
    H, [_, _, a] = Hamiltonian(package, *H_parameters)
    if package=='nanocavity':
        c_ops = no.collapses_tls(H_parameters, VL, VR, kappa, gL, gR, kT, iva=iva)
        L = no.liouvillian(H, list(c_ops))
        S = nme.correlation_AB(L, a.d, a, tlist)
        return S
    elif package=='qutip':
        c_ops = qo.collapses_tls(H_parameters, VL, VR, kappa, gL, gR, kT, iva=iva)
        rho_st= qt.steadystate(H, list(c_ops))
        S = qt.correlation_2op_1t(H=H, state0=rho_st, taulist=tlist, c_ops=list(c_ops), a_op=a.dag(), b_op=a)
        return S


def spectrum(package, H_parameters, VL, VR, kappa, gL, gR, kT, wlist, iva=False, data=False):
    H, [_, _, a] = Hamiltonian(package, *H_parameters)
    if package=='nanocavity-rate':
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
        return E, P, I

    if package=='nanocavity':
        c_ops = collapses('nanocavity', H_parameters, VL, VR, kappa, gL, gR, kT, iva=iva)
        L = no.liouvillian(H, list(c_ops))
        I = kappa * nme.spectrum(L, a, wlist, data=data)
    
    if package=='qutip':
        c_ops = collapses('qutip', H_parameters, VL, VR, kappa, gL, gR, kT, iva=iva)
        I = kappa / (2 * np.pi) \
            * qt.spectrum(H, wlist, list(c_ops), a.dag(), a)
    return I

def g2(package, H_parameters, VL, VR, kappa, gL, gR, kT, tlist, iva=False):
    H, [dg, de, a] = Hamiltonian(package, *H_parameters)
    if package=='nanocavity':
        c_ops = collapses('nanocavity', H_parameters, VL, VR, kappa, gL, gR, kT, iva=iva)
        L = no.liouvillian(H, list(c_ops))
        _, cm = no.collapses(a, H, kT, bath='bosonic', total=False)
        J = no.jump(cm)
        return nme.g2(L, J, tlist)

    elif package=='qutip':
        c_ops = collapses('qutip', H_parameters, VL, VR, kappa, gL, gR, kT, iva=iva)
        rho_st= qt.steadystate(H, list(c_ops))
        g2qt, _ = qt.coherence_function_g2(H, state0=rho_st, taulist=tlist, c_ops=c_ops, a_op=a, solver='me')
        return g2qt

