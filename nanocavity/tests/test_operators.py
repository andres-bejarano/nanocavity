import numpy as np
import nanocavity.operators as no
import nanocavity.rate_equation as nre
from qutip import steadystate, spre, spost, operator_to_vector, vector_to_operator, liouvillian



Eg = 0.4
omegac = 1
delta = 0.9
coupling = 0.3

H_parameters = [Eg, delta, omegac, coupling]

m = 2.5e-2
kappa = 0.1
gL = 1e-3 
gR = 2e-3

VL = 3
VR = -3
kT = 0.1

def test_Htls_nc_QuTiP():
    for rwa in (True, False):
        for n in (1, 2):
            Hnc, [Dg, De, A] = no.H_tls_nc(Eg, delta, omegac, coupling, rwa=rwa, max_bosons=n)
            Hqt, [dg, de, a]  = no.H_tls_QuTiP(Eg, delta, omegac, coupling, rwa=rwa, max_bosons=n)
            
            Enc, _ = Hnc.eigh()
            Eqt, _ = Hqt.eigenstates()
            
            Dg = Dg.toarray()
            De = De.toarray()
            A = A.toarray()
    
            dg = dg.full()
            de = de.full()
            a = a.full()

            assert np.allclose(Enc, Eqt)
            assert np.allclose(Hnc.toarray(), Hqt.full())
            
            assert np.allclose(Dg.T @ Dg + Dg @ Dg.T, np.eye(Dg.shape[0]))
            assert np.allclose(Dg.T @ De + De @ Dg.T, 0)
            assert np.allclose(De.T @ Dg + Dg @ De.T, 0)
            assert np.allclose(De.T @ De + De @ De.T, np.eye(De.shape[0]))
            assert np.allclose(A.T @ A - A @ A.T, a.T @ a - a @ a.T)
            
            assert np.allclose(Dg, dg)
            assert np.allclose(De, de)
            assert np.allclose(A, a)
            


def Nanocav(VL=3, VR=-3, kappa=0.1, m=2.5e-2):
    #nanocav-populations
    Hnc, [dg, de, a] = no.H_tls_nc(Eg, delta, omegac, coupling)
    Enc, Vnc = Hnc.eigh()
    GpL, GmL = nre.transition_rate(Enc, Vnc, [dg, de], gL*np.eye(2), mu=VL, kT=kT)
    GpR, GmR = nre.transition_rate(Enc, Vnc, [dg, de], gR*np.eye(2), mu=VR, kT=kT)
    GL = (GpL + GmL)[:, None]  # VL, VR
    GR = (GpR + GmR)[None, :]
    #damping matrix
    Kp, Km = nre.transition_rate(Enc, Vnc, a, kappa, kT=kT, bath='bosonic')
    K = Kp + Km
    #M direct tunneling
    Mp, Mm = nre.bath_system_bath_rate(Enc, Vnc, a, m, VL, VR, kT)
    #transtion rates matrix
    Gamma = K[np.newaxis, np.newaxis] + GL + GR + Mp + Mm
    return nre.populations(Gamma), Kp, Km, GpL, GmL

def qt_dissipator(VL=3, VR=-3, kappa=0.1, m=2.5e-2):
    Hqt, [dg, de, a] = no.H_tls_QuTiP(Eg, delta, omegac,  coupling)
    Eqt, Vqt = Hqt.eigenstates()
    VLR = VL - VR
    VRL = VR - VL

    #incoherent_evolution
    L = -1.0j * (spre(Hqt.transform(Vqt)) - spost(Hqt.transform(Vqt)))

    #cavity-radiation_bath dissipator
    L +=  kappa * (no.dissipator_bosonic(a, Eqt, Vqt, kT, rate='out') + \
                no.dissipator_bosonic(a.dag(), Eqt, Vqt, kT, rate='in'))

    #molecule-leads dissipator
    L +=  gL * (no.dissipator_fermionic(dg, Eqt, Vqt, VL, kT, rate='out') + \
                no.dissipator_fermionic(de, Eqt, Vqt, VL, kT, rate='out') + \
                no.dissipator_fermionic(dg.dag(), Eqt, Vqt, VL, kT, rate='in') + \
                no.dissipator_fermionic(de.dag(), Eqt, Vqt, VL, kT, rate='in'))

    L += gR * (no.dissipator_fermionic(dg, Eqt, Vqt, VR, kT, rate='out') + \
                no.dissipator_fermionic(de, Eqt, Vqt, VR, kT, rate='out') + \
                no.dissipator_fermionic(dg.dag(), Eqt, Vqt, VR, kT, rate='in') + \
                no.dissipator_fermionic(de.dag(), Eqt, Vqt, VR, kT, rate='in'))
    
    #cavity-leads dissipator
    L += m * (no.dissipator_lead(a, Eqt, Vqt, eV=VLR, kT=kT, rate='out') +\
            no.dissipator_lead(a, Eqt, Vqt, eV=VRL, kT=kT, rate='out') + \
            no.dissipator_lead(a.dag(), Eqt, Vqt, eV=VLR, kT=kT, rate='in') + \
            no.dissipator_lead(a.dag(), Eqt, Vqt, eV=VRL, kT=kT, rate='in'))
    rho_ss = steadystate(L)
    return rho_ss.full().diagonal()

def test_collapses():
    Pnc, _, _, _, _ = Nanocav()
    _, Hqt, c_ops  = no.collapses_tls_QuTiP(H_parameters, VL, VR, kappa, gL, gR, kT, m, lead2lead=True, alone=False)
    _, Vqt = Hqt.eigenstates()
    Pqt = steadystate(Hqt.transform(Vqt), list(c_ops)).full().diagonal()
    assert np.allclose(np.sort(Pnc), np.sort(Pqt))

def test_jump_operator():

    for VL in [-1, 1, 2]:
        for VR in [0, -1, -2]:
            Pnc, Kp, Km, GpL, GmL = Nanocav(VL, VR)
            Ig_nc = nre.photo_current(Kp, Km, Pnc)
            Ie_nc = nre.electro_current(GpL - GmL, Pnc)

            
            [dg, de, a], Hqt, c_ops = no.collapses_tls_QuTiP(H_parameters, VL, VR, kappa, gL, gR, kT, m, lead2lead=True, alone=False)
            
            Eqt, Vqt = Hqt.eigenstates()

            rho_ss = operator_to_vector(steadystate(Hqt.transform(Vqt), list(c_ops)))
            
            J_a_minus = no.jump_bosonic(a, Eqt, Vqt, kT, rate='out') 
            J_a_plus = no.jump_bosonic(a.dag(), Eqt, Vqt, kT, rate='in')        
            Jrho_a_minus = vector_to_operator(J_a_minus * rho_ss)
            Jrho_a_plus = vector_to_operator(J_a_plus * rho_ss)

            J_dgde_plus = no.jump_fermionic(dg.dag(), Eqt, Vqt, VL, kT, rate='in') + \
                    no.jump_fermionic(de.dag(), Eqt, Vqt, VL, kT, rate='in')
            J_dgde_minus = no.jump_fermionic(dg, Eqt, Vqt, VL, kT, rate='out') + \
                    no.jump_fermionic(de, Eqt, Vqt, VL, kT, rate='out')
            
            Jrho_dgde_plus = vector_to_operator(J_dgde_plus * rho_ss)
            Jrho_dgde_minus = vector_to_operator(J_dgde_minus * rho_ss)


            Ig_qt = kappa * (Jrho_a_minus -  Jrho_a_plus).tr()
            Ie_qt = gL * (Jrho_dgde_plus - Jrho_dgde_minus).tr()

            assert np.allclose(Ig_nc, Ig_qt)
            assert np.allclose(Ie_nc, Ie_qt)


def test_dissipators():
    for kappa in [1e-1, 1]:
        for m in [1e-6, 1e-4, 1e-2]:
            Pnc, _, _, _, _ = Nanocav(kappa=kappa, m=m)
            Pqt = qt_dissipator(kappa=kappa, m=m)
            assert np.allclose(np.sort(Pnc), np.sort(Pqt))

def test_Liovillian():
    S_op, Hqt, c_ops = no.collapses_tls_QuTiP(H_parameters, VL, VR, kappa, gL, gR, kT, alone=False)
    Eqt, Vqt = Hqt.eigenstates()
    L1 = no.Liouvillian(Hqt, S_op, VL, VR, kT=kT, gL=gL, gR=gR)
    L2 = liouvillian(Hqt.transform(Vqt), list(c_ops))
    assert np.allclose(L1.full(), L2.full())

