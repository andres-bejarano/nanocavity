import numpy as np
import nanocavity.operators as no
import nanocavity.rate_equation as nre
from qutip import steadystate



Eg = 0.4
omega = 1
delta = 0.9
coupling = 0.3

gammaL = 1e-3 * np.eye(2)
gammaR = 2e-3 * np.eye(2)
kappa =  1
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


def Nanocav(VL, VR, rwa='False'):

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
    M = nre.bath_system_bath_rate(Enc, Vnc, a+a.d, m, VL, VR, kT=kT)
    #transtion rates matrix
    Gamma = K[np.newaxis, np.newaxis] + GL + GR + M
    return nre.populations(Gamma)

def qt(VL, VR, rwa='False'):
    
    glq = gammaL[0, 0]
    grq = gammaR[0, 0]

    Hqt, [dg, de, a] = no.H_tls_QuTiP(Eg, delta, omega,  coupling)
    Eqt, Vqt = Hqt.eigenstates()

    c_g = no.fermionic_collapses(dg, Eqt, Vqt, VL, VR, kT, gL=glq, gR=grq)
    c_e = no.fermionic_collapses(de, Eqt, Vqt, VL, VR, kT, gL=glq, gR=grq)
    c_a = no.bosonic_collapses(a, Eqt, Vqt, kT, kappa) + \
            no.lead_cavity_lead_collapses(a, Eqt, Vqt, VL, VR, kT, m)

    c_ops = c_g + c_e + c_a
    return Hqt, Vqt, c_ops

def test_collapses():
    Pnc = np.sort(Nanocav(VL, VR))
    Hqt, Vqt, c_ops = qt(VL, VR)
    Pqt = steadystate(Hqt.transform(Vqt), c_ops).full().diagonal()
    assert np.allclose(Pnc, np.sort(Pqt))
