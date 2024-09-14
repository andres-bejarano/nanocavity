import numpy as np
import nanocavity.operators as no
import nanocavity.qutip.operators as qo
import nanocavity.rate_equation as nre
import nanocavity.master_equation as nme
import nanocavity.qutip.master_equation as qme
import nanocavity.full_counting as nfc
from scipy.linalg import eig
import qutip as qt
import nanocavity.tls as tls

#system parameters
Eg = 0.4
Delta = 0.9
hw_ph = 1
g_ph = 0.5

H_parameters = Eg, Delta, hw_ph, g_ph 

#bath parameters
Gamma_L = 1e-3
Gamma_R = 2e-3
kappa = 0.1
m = 1e-4
kT = 0.1

def fcs(VL=3, VR=-3):
    #transtion rates, populations and spectrum
    H_parameters = Eg, Delta, hw_ph, g_ph
    Hnc, [Dg, De, A] = tls.Hamiltonian('nanocavity', *H_parameters)
    Enc, Vnc = Hnc.eigh()
    Kp, Km = nre.transition_rate(Enc, Vnc,  A, kappa, kT=kT, bath='bosonic')
    GpL, GmL = nre.transition_rate(Enc, Vnc, [Dg, De], Gamma_L*np.eye(2), mu=VL, kT=kT)
    GpR, GmR = nre.transition_rate(Enc, Vnc, [Dg, De], Gamma_R*np.eye(2), mu=VR, kT=kT)
    Ig_re, Ie_re, Zg_re, Ze_re, _ = nfc.cumulants(Kp, Km, GpL, GmL, GpR, GmR, p=1e-5)
    GL = (GpL + GmL)[:, None]  # VL, VR
    GR = (GpR + GmR)[None, :]
    K = Kp + Km
    return Ig_re, -Ie_re, Zg_re, Ze_re

def test_eig_norm():
    a = 1j * hw_ph - (4 * Gamma_L + kappa) / 2
    b = 1j * Delta  - (4 * Gamma_L) / 2
    c = 1j * g_ph 
    L = np.array([[a, c], [c,  b]])

    E, vl, vr = eig(L, left=True)

    E1, wr = np.linalg.eig(L)
    E2, wl = np.linalg.eig(L.conj().T)

    assert np.allclose(E, E1)
    assert np.allclose(E1, E2.conj())
    
    assert np.allclose(vl, wl)
    assert np.allclose(vr, wr)


    E, vl, vr = nme.eig_norm(L)

    norm = np.einsum("ai,ai->i", vl.conj(), vr) 
    
    assert np.allclose(norm.all(), 1)

def test_stationary():
    for VL, VR in [[3, -3], [2, 0], [-1, 2]]:
        for g_ph in [0.005, 0.05, 0.5]:
            
            H_parameters = Eg, Delta, hw_ph, g_ph
            H, [Dg, De, A] = tls.Hamiltonian('nanocavity', *H_parameters)
            Pme = tls.rho_st('nanocavity', H_parameters, VL, VR, kappa, Gamma_L, Gamma_R, kT)
            Pqt = tls.rho_st('qutip', H_parameters, VL, VR, kappa, Gamma_L, Gamma_R, kT)
            
            assert np.allclose(Pme, Pqt)

def test_correlation_AB():
    tlist = np.linspace(0., 200, 10001)
    for g_ph in [0.005, 0.05, 0.5]:
        Hnc, [_, _, A] = tls.Hamiltonian('nanocavity', Eg, Delta, hw_ph, g_ph)
        Hqt, [_, _, a] = tls.Hamiltonian('qutip', Eg, Delta, hw_ph, g_ph)        
        H_parameters = Eg, Delta, hw_ph, g_ph
        VL, VR = 3, -3
        for iva in (False, True):
            c_nc = tls.collapses('nanocavity', H_parameters, VL, VR, kappa, Gamma_L, Gamma_R, kT, iva=iva)
            L = no.liouvillian(Hnc, list(c_nc))
            Snc = nme.correlation_AB(L, A.d, A, tlist)

            c_qt = tls.collapses('qutip', H_parameters, VL, VR, kappa, Gamma_L, Gamma_R, kT, iva=iva)
            rho_st= qt.steadystate(Hqt, list(c_qt))
            Sqt = qt.correlation_2op_1t(H=Hqt, state0=rho_st, taulist=tlist, c_ops=list(c_qt), a_op=a.dag(), b_op=a)
            assert np.allclose(Snc.real, Sqt.real, atol=1e-5)
            assert np.allclose(Snc.imag, Sqt.imag, atol=1e-5)

def test_sepctrum_vi():
    wlist = np.linspace(0., 1.8, 200)
    hw_vi = 1.9e-1
    g_vi = 1e-8
    U = 2
    H_nc, _, _, anni_list_nc, _ = tls.Hnc_vi(Eg, Delta, hw_ph, g_ph, hw_vi, g_vi, U, [1, 3])
    H_qt, _, _, anni_list_qt = tls.Hqt_vi(Eg, Delta, hw_ph, g_ph, hw_vi, g_vi, U, [1, 3])
    Inc = tls.spectrum_vi('nc', H_nc, anni_list_nc, 3, -3, kappa, Gamma_L, Gamma_R, kT, wlist)
    Iqt = tls.spectrum_vi('qt', H_qt, anni_list_qt, 3, -3, kappa, Gamma_L, Gamma_R, kT, wlist)
    assert np.allclose(Inc, Iqt)
def test_spectrum():
    # wlist = np.linspace(0., 1.8, 100003)
    wlist = np.linspace(0., 1.8, 103)
    Inc = tls.spectrum('nanocavity', H_parameters, 3, -3, kappa, Gamma_L, Gamma_R, kT, wlist)
    Iqt = tls.spectrum('qutip', H_parameters, 3, -3, kappa, Gamma_L, Gamma_R, kT, wlist)
    assert np.allclose(Inc, Iqt)

def test_current():
    for VL, VR in [[3, -3], [2, 0], [-1, 2]]:
            [dg, de, a], Hqt, c_ops = tls.collapses('qutip', H_parameters, VL, VR, kappa, Gamma_L, Gamma_R, kT, alone=False)

            #left electrode
            cp_gL, cm_gL = qo.collapses( dg, Hqt, kT, bath='fermionic', mu=VL, total=False)
            cp_eL, cm_eL = qo.collapses(de, Hqt, kT, bath='fermionic', mu=VL, total=False)
            CpL = list(np.sqrt(Gamma_L) * np.array(cp_gL + cp_eL))
            CmL = list(np.sqrt(Gamma_L) * np.array(cm_gL + cm_eL))

            #cavity mode
            cap, cam = qo.collapses(a, Hqt, kT, bath='bosonic', total=False)
            Cp = list(np.sqrt(kappa) * np.array(cap))
            Cm = list(np.sqrt(kappa) * np.array(cam))
            
            L = qo.liouvillian(Hqt, list(c_ops))

            Ig_me = qme.current(qo.jump(Cm) - qo.jump(Cp), L)
            Ie_me = qme.current(qo.jump(CpL) - qo.jump(CmL), L)
            Ig_re, Ie_re, _, _ = fcs(VL, VR) 
            
            #assert np.allclose(Ig_me, Ig_re)
            #assert np.allclose(Ie_me, Ie_re)

def test_noise():
    for VL, VR in [[-3, 0], [-2.1, 1], [0, 3]]:
            [dg, de, a], Hqt, c_ops = tls.collapses('qutip', H_parameters, VL, VR, kappa, Gamma_L, Gamma_R, kT, alone=False)

            #left electrode
            cp_gL, cm_gL = qo.collapses(dg, Hqt, kT, bath='fermionic', mu=VL, total=False)
            cp_eL, cm_eL = qo.collapses(de, Hqt, kT, bath='fermionic', mu=VL, total=False)
            CpL = list(np.sqrt(Gamma_L) * np.array(cp_gL + cp_eL))
            CmL = list(np.sqrt(Gamma_L) * np.array(cm_gL + cm_eL))
            
            #cavity mode
            cap, cam = qo.collapses(a, Hqt, kT, bath='bosonic', total=False)
            Cp = list(np.sqrt(kappa) * np.array(cap))
            Cm = list(np.sqrt(kappa) * np.array(cam))

                
            L = qo.liouvillian(Hqt, list(c_ops))
            Zg_me = qme.noise(L, qo.jump(Cp), qo.jump(Cm))
            Ze_me = qme.noise(L, qo.jump(CpL), qo.jump(CmL))
            _, _, Zg_re, Ze_re = fcs(VL, VR)
            #assert np.allclose(Zg_me, Zg_re, atol=1e-7)
            #assert np.allclose(Ze_me, Ze_re, atol=1e-6)


def test_g2():
    VL, VR = 10, -10
    g_ph = 0.005 
    Delta = 0.99
    H_parameters = Eg, Delta, hw_ph, g_ph
    tlist = np.linspace(0., 3000, 10)
    g2nc = tls.g2('nanocavity', H_parameters, VL, VR, kappa, Gamma_L, Gamma_R, kT, tlist)
    g2qt = tls.g2('qutip', H_parameters, VL, VR, kappa, Gamma_L, Gamma_R, kT, tlist)
    assert np.allclose(g2nc, g2qt, atol=1e-1)

