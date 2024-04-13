import numpy as np
import nanocavity.operators as no
import nanocavity.rate_equation as nre
import nanocavity.master_equation as nme
import nanocavity.full_counting as nfc
from qutip import steadystate


#system parameters
Eg = -0.23
delta = 0.9
omegac = 1
coupling = 3.5e-3
u = 1.2

#bath parameters
gL = 1e-3
gR = 2e-3
kappa = 0.1
m = 0
kT = 0.1

H, [dg, de, a] = no.H_tls_QuTiP(Eg, delta, omegac,  coupling)
H += u * dg.dag() * de.dag() * de * dg
E, V =  H.eigenstates()

#Liouvillian
def Li(VL=3, VR=-3):
    L = no.Liouvillian(H, [dg, de, a], VL, VR, kT=kT,gL=gL, gR=gR)
    return L

def fcs(VL=3, VR=-3):
    Hnc, [Dg, De, A] = no.H_tls_nc(Eg, delta, omegac, coupling)
    Hnc += u * Dg.d * De.d * De * Dg
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
    return Ig_re, Ie_re, Zg_re, Ze_re

def test_current():
    for VL in [-3, -2.1, -1.1, 2]:
        for VR in [-1.3, 1.4, 2.5]:
            
            VLR = VL - VR
            VRL = VR - VL
            L = Li(VL, VR)

            Jtip_out = gL * no.jump_fermionic(dg.dag(), E, V, VL, kT, rate='in') + \
                       gL * no.jump_fermionic(de.dag(), E, V, VL, kT, rate='in') + \
                        m * no.dissipator_lead(a, E, V, eV=VLR, kT=kT, rate='out') + \
                        m * no.dissipator_lead(a.dag(), E, V, eV=VLR, kT=kT, rate='in')


            Jtip_in = gL * no.jump_fermionic(dg, E, V, VL, kT, rate='out') + \
                      gL * no.jump_fermionic(de, E, V, VL, kT, rate='out') + \
                       m * no.dissipator_lead(a, E, V, eV=VRL, kT=kT, rate='out') + \
                       m * no.dissipator_lead(a.dag(), E, V, eV=VRL, kT=kT, rate='in')

            Ja_out = kappa * no.jump_bosonic(a, E, V, kT, rate='out')
            Ja_in = kappa * no.jump_bosonic(a.dag(), E, V, kT, rate='in')

            Ig_me = nme.current(Ja_out - Ja_in, L)
            Ie_me = nme.current(Jtip_in - Jtip_out, L)
    
            Ig_re, Ie_re, _, _ = fcs(VL, VR)

            assert np.allclose(Ig_me, Ig_re)
            assert np.allclose(Ie_me, Ie_re)

def test_cumulants():
   #nme.cumulants is too slow we avoid to sweep the bias    
    Ig_re, Ie_re, Zg_re, Ze_re = fcs(3, -3)
    Ig_me, Ie_me, Zg_me, Ze_me = nme.cumulants(H, [dg, de, a], 3, -3, kT=kT, gL=gL, gR=gR) 
    assert np.allclose(Ig_me, Ig_re)
    assert np.allclose(Ie_me, Ie_re)
    assert np.allclose(Zg_me, Zg_re, atol=1e-7)
    assert np.allclose(Ze_me, Ze_re, atol=1e-6)

def test_noise():
    Ja_out = kappa * no.jump_bosonic(a, E, V, kT, rate='out')
    Ja_in = kappa * no.jump_bosonic(a.dag(), E, V, kT, rate='in')
    
    for VL in [-3, -2.1, -1.1, 2]:
        for VR in [-1.3, 1.4, 2.5]:
            L = Li(VL, VR)
            
            Jtip_in = gL * no.jump_fermionic(dg.dag(), E, V, VL, kT, rate='in') + \
                      gL * no.jump_fermionic(de.dag(), E, V, VL, kT, rate='in')
            
            Jtip_out = gL * no.jump_fermionic(dg, E, V, VL, kT, rate='out') + \
                       gL * no.jump_fermionic(de, E, V, VL, kT, rate='out')
            
            Zg_me = nme.noise(L, Ja_in, Ja_out)
            Ze_me = nme.noise(L, Jtip_in, Jtip_out)
            _, _, Zg_re, Ze_re = fcs(VL, VR)
            assert np.allclose(Zg_me, Zg_re, atol=1e-7)
            assert np.allclose(Ze_me, Ze_re, atol=1e-7)
