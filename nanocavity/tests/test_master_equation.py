import numpy as np
import nanocavity.operators as no
import nanocavity.qutip.operators as qo
import nanocavity.rate_equation as nre
import nanocavity.master_equation as nme
import nanocavity.qutip.master_equation as qme
import nanocavity.full_counting as nfc
from scipy.linalg import eig
import qutip as qt

#system parameters
Eg = 0.4
delta = 0.9
omegac = 1
coupling = 0.5

H_parameters = Eg, delta, omegac, coupling 

#bath parameters
gL = 1e-3
gR = 2e-3
kappa = 0.1
m = 1e-4
kT = 0.1

Hqt, [dg, de, a] = qo.H_tls(Eg, delta, omegac, coupling)
Eqt, Vqt =  Hqt.eigenstates()


Hnc, [Dg, De, A] = no.H_tls(Eg, delta, omegac, coupling)
Enc, Vnc = Hnc.eigh()

def system(pack):
    if pack=='nanocavity':
        return no.H_tls(Eg, delta, omegac, coupling)
    elif pack=='qutip':
        return qo.H_tls(Eg, delta, omegac, coupling)

def fcs(VL=3, VR=-3):
    #transtion rates, populations and spectrum
    Hnc, [Dg, De, A] = no.H_tls(Eg, delta, omegac, coupling)
    Enc, Vnc = Hnc.eigh()
    Kp, Km = nre.transition_rate(Enc, Vnc,  A, kappa, kT=kT, bath='bosonic')
    GpL, GmL = nre.transition_rate(Enc, Vnc, [Dg, De], gL*np.eye(2), mu=VL, kT=kT)
    GpR, GmR = nre.transition_rate(Enc, Vnc, [Dg, De], gR*np.eye(2), mu=VR, kT=kT)
    Ig_re, Ie_re, Zg_re, Ze_re, _ = nfc.cumulants(Kp, Km, GpL, GmL, GpR, GmR, p=1e-5)
    GL = (GpL + GmL)[:, None]  # VL, VR
    GR = (GpR + GmR)[None, :]
    K = Kp + Km
    Gamma = K[np.newaxis, np.newaxis] + GL + GR
    return Ig_re, -Ie_re, Zg_re, Ze_re

def test_eig_norm():
    a = 1j * omegac - (4 * gL + kappa) / 2
    b = 1j * delta  - (4 * gL) / 2
    c = 1j * coupling 
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

def test_correlation_AB():
    tlist = np.linspace(0., 200, 10001)
    for coupling in [0.005, 0.05, 0.5]:
        Hnc, [Dg, De, A] = system('nanocavity')
        Hqt, [dg, de, a] = system('qutip')        
        H_parameters = Eg, delta, omegac, coupling
        VL, VR = 3, -3
        for iva in (False, True):
            c_nc = no.collapses_tls(H_parameters, VL, VR, kappa, gL, gR, kT, iva=iva)
            L = no.liouvillian(Hnc.toarray(), list(c_nc))
            Snc = nme.correlation_AB(L, A.d, A, tlist)

            c_qt = qo.collapses_tls(H_parameters, VL, VR, kappa, gL, gR, kT, iva=iva)
            rho_st= qt.steadystate(Hqt, list(c_qt))
            Sqt = qt.correlation_2op_1t(H=Hqt, state0=rho_st, taulist=tlist, c_ops=list(c_qt), a_op=a.dag(), b_op=a)
            print(iva, coupling)
            assert np.allclose(Snc.real, Sqt.real, atol=1e-5)
            assert np.allclose(Snc.imag, Sqt.imag, atol=1e-5)

def test_spectrum():
    wlist = np.linspace(0., 1.8, 100003)
    Inc = nme.spectrum_tls('nanocavity', H_parameters, 3, -3, kappa, gL, gR, kT, wlist)
    Iqt = nme.spectrum_tls('qutip', H_parameters, 3, -3, kappa, gL, gR, kT, wlist)
    assert np.allclose(Inc, Iqt)

def test_current():
    for VL, VR in [[3, -3], [2, 0], [-1, 2]]:
            [dg, de, a], Hqt, c_ops = qo.collapses_tls(H_parameters, VL, VR, kappa, gL, gR, kT, alone=False)
            _, Vqt = Hqt.eigenstates()


            #left electrode
            cp_gL, cm_gL = qo.collapses(dg, Hqt, kT, bath='fermionic', mu=VL, total=False)
            cp_eL, cm_eL = qo.collapses(de, Hqt, kT, bath='fermionic', mu=VL, total=False)
            CpL = list(np.sqrt(gL) * np.array(cp_gL + cp_eL))
            CmL = list(np.sqrt(gL) * np.array(cm_gL + cm_eL))

            #cavity mode
            cap, cam = qo.collapses(a, Hqt, kT, bath='bosonic', total=False)
            Cp = list(np.sqrt(kappa) * np.array(cap))
            Cm = list(np.sqrt(kappa) * np.array(cam))
            
            L = qo.liouvillian(Hqt, list(c_ops))

            Ig_me = qme.current(qo.jump(Cm) - qo.jump(Cp), L)
            Ie_me = qme.current(qo.jump(CpL) - qo.jump(CmL), L)
            Ig_re, Ie_re, _, _ = fcs(VL, VR) 
            
            assert np.allclose(Ig_me, Ig_re)
            assert np.allclose(Ie_me, Ie_re)

def test_noise():
    for VL, VR in [[-3, 0], [-2.1, 1], [0, 3]]:
            [dg, de, a], Hqt, c_ops = qo.collapses_tls(H_parameters, VL, VR, kappa, gL, gR, kT, alone=False)
            _, Vqt = Hqt.eigenstates()


            #left electrode
            cp_gL, cm_gL = qo.collapses(dg, Hqt, kT, bath='fermionic', mu=VL, total=False)
            cp_eL, cm_eL = qo.collapses(de, Hqt, kT, bath='fermionic', mu=VL, total=False)
            CpL = list(np.sqrt(gL) * np.array(cp_gL + cp_eL))
            CmL = list(np.sqrt(gL) * np.array(cm_gL + cm_eL))
            
            #cavity mode
            cap, cam = qo.collapses(a, Hqt, kT, bath='bosonic', total=False)
            Cp = list(np.sqrt(kappa) * np.array(cap))
            Cm = list(np.sqrt(kappa) * np.array(cam))

                
            L = qo.liouvillian(Hqt, list(c_ops))
            Zg_me = qme.noise(L, qo.jump(Cp), qo.jump(Cm))
            Ze_me = qme.noise(L, qo.jump(CpL), qo.jump(CmL))
            _, _, Zg_re, Ze_re = fcs(VL, VR)
            assert np.allclose(Zg_me, Zg_re, atol=1e-7)
            assert np.allclose(Ze_me, Ze_re, atol=1e-7)


