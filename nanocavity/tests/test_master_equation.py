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
kT = 1e-2
VL = 3
VR = -3

Hqt, [dg, de, a] = no.H_tls_QuTiP(Eg, delta, omegac,  coupling)
Hqt += u * dg.dag() * de.dag() * de * dg
Eqt, Vqt =  Hqt.eigenstates()
L = no.Liouvillian(Hqt, [dg, de, a], VL, VR, gL=gL, gR=gR)

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
    VLR = VL - VR
    VRL = VR - VL
    H = Hqt
    E, V = H.eigenstates()

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
    
    Ig_re, Ie_re, _, _ = fcs()

    assert np.allclose(Ig_me, Ig_re)
    assert np.allclose(Ie_me, Ie_re)

def test_cumulants():
    Ig_re, Ie_re, Zg_re, Ze_re = fcs()
    Ig_me, Ie_me, Zg_me, Ze_me = nme.cumulants(Hqt, [dg, de, a], VL, VR, gL=gL, gR=gR) 
    assert np.allclose(Ig_me, Ig_re)
    assert np.allclose(Ie_me, Ie_re)
    assert np.allclose(Zg_me, Zg_re, atol=1e-7)
    assert np.allclose(Ze_me, Ze_re)

def test_noise():
    Ja_out = kappa * no.jump_bosonic(a, Eqt, Vqt, kT, rate='out')
    Ja_in = kappa * no.jump_bosonic(a.dag(), Eqt, Vqt, kT, rate='in')
    Zg_me = nme.noise(L, Ja_in, Ja_out, wlist=[0])
    _, _, Zg_re, Ze_re = fcs()
    assert np.allclose(Zg_me, Zg_re, atol=1e-7)

