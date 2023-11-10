import numpy as np
from nanocavity.distributions import *
from nanocavity.operators import *
from nanocavity.rate_equation import transition_rate, transition_rate_matrix, populations
from secondquant.composite import *
from qutip import steadystate


def test_Htls_nc_QuTiP():
    Eg = 0.4
    omega = 1
    delta = 0.9
    coupling = 0.3
    
    Hnc, _, _ = H_tls_nc(Eg, delta, omega, coupling)
    Hqt,_ = H_tls_QuTiP(Eg, delta, omega, coupling)

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
    Hnc, La, Ld = H_tls_nc(Eg, delta, omega, coupling)
    Enc, Vnc = Hnc.eigh()
    GpL, GmL = transition_rate(Enc, Vnc, Ld, gamma, mu=VL, kT=kT)
    GpR, GmR = transition_rate(Enc, Vnc, Ld, gamma, mu=VR, kT=kT)

    GL, GR = transition_rate_matrix(GpL + GmL, GpR + GmR)

    #damping matrix
    Kp, Km = transition_rate(Enc, Vnc, La, kappa, kT=kT, bath='bosonic')
    K = Kp + Km

    #transtion rates matrix
    Gamma = K[np.newaxis, np.newaxis] + GL + GR
    Pnc = np.sort(populations(Gamma))

    #QuTiP populations

    Hqt, L = H_tls_QuTiP(Eg, delta, omega,  coupling)
    Eqt, Vqt = Hqt.eigenstates()
    collapses_cavity = bosonic_collapses(L[0], E=Eqt, V=Vqt, kT=kT, k=kappa)
    collapses_ground = fermionic_collapses(L[1], E=Eqt, V=Vqt, VL=VL, VR=VR, kT=1e-2, g=gamma[0, 0])
    collapses_excited = fermionic_collapses(L[2], E=Eqt, V=Vqt, VL=VL, VR=VR, kT=1e-2 ,g=gamma[0, 0])
    c_ops = collapses_cavity + collapses_ground + collapses_excited
    A = steadystate(Hqt, c_ops).full()
    Pqt = np.linalg.eigh(A)[0]

    assert np.allclose(Pnc, Pqt)
