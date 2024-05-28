import numpy as np
import nanocavity.operators as no
import nanocavity.rate_equation as nre
import nanocavity.master_equation as nme
import nanocavity.full_counting as nfc
from qutip import steadystate, liouvillian


#system parameters
Eg = 0.4
delta = 0.9
omegac = 1
coupling = 0.3

H_parameters = Eg, delta, omegac, coupling 

#bath parameters
gL = 1e-3
gR = 2e-3
kappa = 0.1
kT = 0.1


def fcs(VL=3, VR=-3):
    Hnc, [Dg, De, A] = no.H_tls_nc(Eg, delta, omegac, coupling)
    Enc, Vnc = Hnc.eigh()

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
            [dg, de, a], Hqt, c_ops = no.collapses_tls_QuTiP(H_parameters, VL, VR, kappa, gL, gR, kT, alone=False)
            _, Vqt = Hqt.eigenstates()


            #left electrode
            cp_gL, cm_gL = no.collapses(dg, Hqt, kT, bath='fermionic', mu=VL, total=False)
            cp_eL, cm_eL = no.collapses(de, Hqt, kT, bath='fermionic', mu=VL, total=False)
            CpL = list(np.sqrt(gL) * np.array(cp_gL + cp_eL))
            CmL = list(np.sqrt(gL) * np.array(cm_gL + cm_eL))

            #cavity mode
            cap, cam = no.collapses(a, Hqt, kT, bath='bosonic', total=False)
            Cp = list(np.sqrt(kappa) * np.array(cap))
            Cm = list(np.sqrt(kappa) * np.array(cam))
            
            L = no.liouvillian(Hqt.transform(Vqt), list(c_ops))

            Ig_me = nme.current(no.jump(Cm) - no.jump(Cp), L)
            Ie_me = nme.current(no.jump(CpL) - no.jump(CmL), L)
    
            Ig_re, Ie_re, _, _ = fcs(VL, VR)

            assert np.allclose(Ig_me, Ig_re)
            assert np.allclose(Ie_me, Ie_re)

def test_noise():
    for VL, VR in [[-3, 0], [-2.1, 1], [0, 3]]:
            [dg, de, a], Hqt, c_ops = no.collapses_tls_QuTiP(H_parameters, VL, VR, kappa, gL, gR, kT, alone=False)
            _, Vqt = Hqt.eigenstates()


            #left electrode
            cp_gL, cm_gL = no.collapses(dg, Hqt, kT, bath='fermionic', mu=VL, total=False)
            cp_eL, cm_eL = no.collapses(de, Hqt, kT, bath='fermionic', mu=VL, total=False)
            CpL = list(np.sqrt(gL) * np.array(cp_gL + cp_eL))
            CmL = list(np.sqrt(gL) * np.array(cm_gL + cm_eL))
            
            #cavity mode
            cap, cam = no.collapses(a, Hqt, kT, bath='bosonic', total=False)
            Cp = list(np.sqrt(kappa) * np.array(cap))
            Cm = list(np.sqrt(kappa) * np.array(cam))

                
            L = no.liouvillian(Hqt.transform(Vqt), list(c_ops))
            Zg_me = nme.noise(L, no.jump(Cp), no.jump(Cm))
            Ze_me = nme.noise(L, no.jump(CpL), no.jump(CmL))
            _, _, Zg_re, Ze_re = fcs(VL, VR)
            assert np.allclose(Zg_me, Zg_re, atol=1e-7)
            assert np.allclose(Ze_me, Ze_re, atol=1e-7)
