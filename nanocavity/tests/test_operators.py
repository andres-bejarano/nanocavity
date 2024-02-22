import numpy as np
import nanocavity.operators as no
import nanocavity.rate_equation as nre
from qutip import steadystate, spre, spost



Eg = 0.4
omega = 1
delta = 0.9
coupling = 0.3

gammaL = 1e-3 * np.eye(2)
gammaR = 2e-3 * np.eye(2)
kappa =  0.1
kT = 0.1
VL = 3
VR = -3
m = 2.5e-2

def test_Htls_nc_QuTiP():
    for rwa in (True, False):
        for n in (1, 2):
            Hnc, _ = no.H_tls_nc(Eg, delta, omega, coupling, rwa=rwa, max_bosons=n)
            Hqt,_ = no.H_tls_QuTiP(Eg, delta, omega, coupling, rwa=rwa, max_bosons=n)
            Enc, _ = Hnc.eigh()
            Eqt, _ = Hqt.eigenstates()
            assert np.allclose(Enc, Eqt)


def Nanocav(VL, VR):

    #nanocav-populations
    Hnc, [dg, de, a] = no.H_tls_nc(Eg, delta, omega, coupling)
    Enc, Vnc = Hnc.eigh()
    GpL, GmL = nre.transition_rate(Enc, Vnc, [dg, de], gammaL, mu=VL, kT=kT)
    GpR, GmR = nre.transition_rate(Enc, Vnc, [dg, de], gammaR, mu=VR, kT=kT)
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

def qt(VL, VR):
    
    glq = gammaL[0, 0]
    grq = gammaR[0, 0]

    Hqt, [dg, de, a] = no.H_tls_QuTiP(Eg, delta, omega,  coupling)
    Eqt, Vqt = Hqt.eigenstates()

    c_g = no.fermionic_collapses(dg, Eqt, Vqt, VL, VR, kT, glq, grq)
    c_e = no.fermionic_collapses(de, Eqt, Vqt, VL, VR, kT, glq, grq)
    c_a = no.bosonic_collapses(a, Eqt, Vqt, kT, kappa) + \
            no.lead_cavity_lead_collapses(a, Eqt, Vqt, VL, VR, kT, m)

    c_ops = c_g + c_e + c_a
    L = [dg, de, a]

    return Hqt, Vqt, Eqt, c_ops, L

def qt_dissipator():
    Hqt, [dg, de, a] = no.H_tls_QuTiP(Eg, delta, omega,  coupling)
    Eqt, Vqt = Hqt.eigenstates()
    VLR = VL - VR
    VRL = VR - VL
    glq = gammaL[0, 0]
    grq = gammaR[0, 0]

    #incoherent_evolution
    L = -1.0j * (spre(Hqt.transform(Vqt)) - spost(Hqt.transform(Vqt)))

    #cavity-radiation_bath dissipator
    L +=  kappa * (no.dissipator_bosonic(a, Eqt, Vqt, kT, rate='out') + \
                no.dissipator_bosonic(a.dag(), Eqt, Vqt, kT, rate='in'))

    #molecule-leads dissipator
    L +=  glq * (no.dissipator_fermionic(dg, Eqt, Vqt, VL, kT, rate='out') + \
                no.dissipator_fermionic(de, Eqt, Vqt, VL, kT, rate='out') + \
                no.dissipator_fermionic(dg.dag(), Eqt, Vqt, VL, kT, rate='in') + \
                no.dissipator_fermionic(de.dag(), Eqt, Vqt, VL, kT, rate='in'))

    L += grq * (no.dissipator_fermionic(dg, Eqt, Vqt, VR, kT, rate='out') + \
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
    Pnc, _, _, _, _ = Nanocav(VL, VR)
    Hqt, Vqt, _, c_ops, _ = qt(VL, VR)
    Pqt = steadystate(Hqt.transform(Vqt), c_ops).full().diagonal()
    assert np.allclose(np.sort(Pnc), np.sort(Pqt))

def test_jump_operator():

    for VL in [-1, 1, 2]:
        for VR in [0, -1, -2]:
            Pnc, Kp, Km, GpL, GmL = Nanocav(VL, VR)
            Ig_nc = nre.photo_current(Kp, Km, Pnc)
            Ie_nc = nre.electro_current(GpL - GmL, Pnc)

            Hqt, Vqt, Eqt, c_ops, L = qt(VL, VR)
            [dg, de, a] = L

            rho_ss = steadystate(Hqt.transform(Vqt), c_ops)
    
            Jrho_a = no.jump_bosonic(a, rho_ss, Eqt, Vqt, kT, rate='out') - \
                    no.jump_bosonic(a.dag(), rho_ss, Eqt, Vqt, kT, rate='in')
    
            Jrho_dgde_plus = no.jump_fermionic(dg.dag(), rho_ss, Eqt, Vqt, VL, kT, rate='in') + \
                    no.jump_fermionic(de.dag(), rho_ss, Eqt, Vqt, VL, kT, rate='in')
                    
            Jrho_dgde_minus = no.jump_fermionic(dg, rho_ss, Eqt, Vqt, VL, kT, rate='out') + \
                    no.jump_fermionic(de, rho_ss, Eqt, Vqt, VL, kT, rate='out')
    
            Ig_qt = kappa * Jrho_a.tr()
            Ie_qt = gammaL[0, 0] * (Jrho_dgde_plus - Jrho_dgde_minus).tr()
    
            assert np.allclose(Ig_nc, Ig_qt)
            assert np.allclose(Ie_nc, Ie_qt)



def test_dissipators():
    for kappa in [1e-1, 1]:
        for m in [1e-6, 1e-4, 1e-2]:
            Pnc, _, _, _, _ = Nanocav(VL, VR)
            Pqt = qt_dissipator()
            assert np.allclose(np.sort(Pnc), np.sort(Pqt))

