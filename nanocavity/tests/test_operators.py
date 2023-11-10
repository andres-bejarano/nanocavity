import numpy as np
import nanocavity.operators as no
import nanocavity.rate_equation as nre
from qutip import steadystate


def test_Htls_nc_QuTiP():
    Eg = 0.4
    omega = 1
    delta = 0.9
    coupling = 0.3
    
    Hnc, _, _ = no.H_tls_nc(Eg, delta, omega, coupling)
    Hqt,_ = no.H_tls_QuTiP(Eg, delta, omega, coupling)

    Enc, _ = Hnc.eigh()
    Eqt, _ = Hqt.eigenstates()
    assert np.allclose(Enc, Eqt)

def test_collapses():
    Eg = 0.4
    delta = 0.9
    omega = 1
    coupling= 0.3

    gamma = 1e-3 * np.eye(2)
    kappa =  1
    kT = 1e-2
    VL = 3
    VR = -3

    #nanocav-populations
    Hnc, La, Ld = no.H_tls_nc(Eg, delta, omega, coupling)
    Enc, Vnc = Hnc.eigh()
    GpL, GmL = nre.transition_rate(Enc, Vnc, Ld, gamma, mu=VL, kT=kT)
    GpR, GmR = nre.transition_rate(Enc, Vnc, Ld, gamma, mu=VR, kT=kT)

    GL, GR = nre.transition_rate_matrix(GpL + GmL, GpR + GmR)

    #damping matrix
    Kp, Km = nre.transition_rate(Enc, Vnc, La, kappa, kT=kT, bath='bosonic')
    K = Kp + Km

    #transtion rates matrix
    Gamma = K[np.newaxis, np.newaxis] + GL + GR
    Pnc = np.sort(nre.populations(Gamma))

    #QuTiP populations

    Hqt, L = no.H_tls_QuTiP(Eg, delta, omega,  coupling)
    Eqt, Vqt = Hqt.eigenstates()
    collapses_cavity = no.bosonic_collapses(L[0], E=Eqt, V=Vqt, kT=kT, k=kappa)
    collapses_ground = no.fermionic_collapses(L[1], E=Eqt, V=Vqt, VL=VL, VR=VR, kT=1e-2, g=gamma[0, 0])
    collapses_excited = no.fermionic_collapses(L[2], E=Eqt, V=Vqt, VL=VL, VR=VR, kT=1e-2 ,g=gamma[0, 0])
    c_ops = collapses_cavity + collapses_ground + collapses_excited
    A = steadystate(Hqt, c_ops).full()
    Pqt = np.linalg.eigh(A)[0]

    assert np.allclose(Pnc, Pqt)
