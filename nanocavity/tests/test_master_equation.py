import numpy as np
import nanocavity.operators as no
import nanocavity.qutip.operators as qo
import nanocavity.rate_equation as nre
import nanocavity.master_equation as nme
import nanocavity.qutip.master_equation as qme
import nanocavity.full_counting as nfc
from qutip import steadystate, liouvillian


#system parameters
Eg = 0.4
delta = 0.9
omegac = 1

H_parameters = Eg, delta, omegac, coupling 
coupling = 0.5
u = 1.2

#bath parameters
gL = 1e-3
gR = 2e-3
kappa = 0.1
m = 1e-4
kT = 0.1


def fcs(VL=3, VR=-3):
    #transtion rates, populations and spectrum
    Kp, Km = nre.transition_rate(Enc, Vnc,  A, kappa, kT=kT, bath='bosonic')
    GpL, GmL = nre.transition_rate(Enc, Vnc, [Dg, De], gL*np.eye(2), mu=VL, kT=kT)
    GpR, GmR = nre.transition_rate(Enc, Vnc, [Dg, De], gR*np.eye(2), mu=VR, kT=kT)
    Ig_re, Ie_re, Zg_re, Ze_re, _ = nfc.cumulants(Kp, Km, GpL, GmL, GpR, GmR, p=1e-5)
    GL = (GpL + GmL)[:, None]  # VL, VR
    GR = (GpR + GmR)[None, :]
    K = Kp + Km
    Gamma = K[np.newaxis, np.newaxis] + GL + GR
    P_re = nre.populations(Gamma)
    return Ig_re, -Ie_re, Zg_re, Ze_re

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
            
            L = qo.liouvillian(Hqt.transform(Vqt), list(c_ops))

            Ig_me = qme.current(qo.jump(Cm) - qo.jump(Cp), L)
            Ie_me = qme.current(qo.jump(CpL) - qo.jump(CmL), L)
    
    return P_re, Ig_re, Ie_re, Zg_re, Ze_re

def test_eigenoperator():
    for AA in [Dg, De, A]:
        #if we sum all eigenoperators we need to recover Dg
        #ij are the matrix elements of each eigenoperator
        A_eigen = nme.eigen_operator(AA.toarray(), Vnc)
        A_eigen_sum = np.einsum('ijkp->ij', A_eigen)
        assert np.allclose(A_eigen_sum, AA.toarray())


def test_Liouvillian():
    H1 = Hnc - u * Dg.d * De.d * De * Dg
    H2 = Hqt -u * dg.dag() * de.dag() * de * dg

    for meth in ('einsum', 'kron'):
        Lnc = nme.liouvillian(H1, [Dg, De, A], gL, gR, kappa, method=meth)
        Lqt = no.Liouvillian(H2, [dg, de, a], 10, -10, kT=1e-2,gL=gL, gR=gR)

        Enc, _ = np.linalg.eig(Lnc)
        Eqt, _ = np.linalg.eig(Lqt.full())

        E1real = np.sort(Enc.real)
        E2real = np.sort(Eqt.real)

        E1imag = np.sort(Enc.imag)
        E2imag = np.sort(Eqt.imag)

        E1abs = np.sort(abs(Enc))
        E2abs = np.sort(abs(Eqt))

        for i in range(len(Enc)):
            assert np.allclose(E1real, E2real)
            assert np.allclose(E1imag, E2imag)
            assert np.allclose(E1abs, E2abs)

test_Liouvillian()

def test_current():
    for VL in [-3, -2.1, -1.1, 2]:
        for VR in [-1.3, 1.4, 2.5]:

            VLR = VL - VR
            VRL = VR - VL
            L = Li(VL, VR)

            Jtip_out = gL * no.jump_fermionic(dg.dag(), Eqt, Vqt, VL, kT, rate='in') + \
                       gL * no.jump_fermionic(de.dag(), Eqt, Vqt, VL, kT, rate='in') + \
                        m * no.dissipator_lead(a, Eqt, Vqt, eV=VLR, kT=kT, rate='out') + \
                        m * no.dissipator_lead(a.dag(), Eqt, Vqt, eV=VLR, kT=kT, rate='in')


            Jtip_in = gL * no.jump_fermionic(dg, Eqt, Vqt, VL, kT, rate='out') + \
                      gL * no.jump_fermionic(de, Eqt, Vqt, VL, kT, rate='out') + \
                       m * no.dissipator_lead(a, Eqt, Vqt, eV=VRL, kT=kT, rate='out') + \
                       m * no.dissipator_lead(a.dag(), Eqt, Vqt, eV=VRL, kT=kT, rate='in')

            Ja_out = kappa * no.jump_bosonic(a, Eqt, Vqt, kT, rate='out')
            Ja_in = kappa * no.jump_bosonic(a.dag(), Eqt, Vqt, kT, rate='in')

            Ig_me = nme.current(Ja_out - Ja_in, L)
            Ie_me = nme.current(Jtip_in - Jtip_out, L)

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

                
            L = qo.liouvillian(Hqt.transform(Vqt), list(c_ops))
            Zg_me = qme.noise(L, qo.jump(Cp), qo.jump(Cm))
            Ze_me = qme.noise(L, qo.jump(CpL), qo.jump(CmL))
            _, _, Zg_re, Ze_re = fcs(VL, VR)
            assert np.allclose(Zg_me, Zg_re, atol=1e-7)
            assert np.allclose(Ze_me, Ze_re, atol=1e-7)
def test_cumulants():
   #nme.cumulants is too slow we avoid to sweep the bias
    _, Ig_re, Ie_re, Zg_re, Ze_re = fcs(3, -3)
    Ig_me, Ie_me, Zg_me, Ze_me = nme.cumulants(Hqt, [dg, de, a], 3, -3, kT=kT, gL=gL, gR=gR)
    assert np.allclose(Ig_me, Ig_re)
    assert np.allclose(Ie_me, Ie_re)
    #assert np.allclose(Zg_me, Zg_re, atol=1e-7)
    #assert np.allclose(Ze_me, Ze_re, atol=1e-6)

def test_noise():
    Ja_out = kappa * no.jump_bosonic(a, Eqt, Vqt, kT, rate='out')
    Ja_in = kappa * no.jump_bosonic(a.dag(), Eqt, Vqt, kT, rate='in')

    for VL in [-3, -2.1, -1.1, 2]:
        for VR in [-1.3, 1.4, 2.5]:
            L = Li(VL, VR)

            Jtip_in = gL * no.jump_fermionic(dg.dag(), Eqt, Vqt, VL, kT, rate='in') + \
                      gL * no.jump_fermionic(de.dag(), Eqt, Vqt, VL, kT, rate='in')

            Jtip_out = gL * no.jump_fermionic(dg, Eqt, Vqt, VL, kT, rate='out') + \
                       gL * no.jump_fermionic(de, Eqt, Vqt, VL, kT, rate='out')

            Zg_me = nme.noise(L, Ja_in, Ja_out)
            Ze_me = nme.noise(L, Jtip_in, Jtip_out)
            _, _, _, Zg_re, Ze_re = fcs(VL, VR)
            assert np.allclose(Zg_me, Zg_re, atol=1e-6)
            assert np.allclose(Ze_me, Ze_re, atol=1e-6)
