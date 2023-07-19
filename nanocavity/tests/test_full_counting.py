import numpy as np
from secondquant.composite import *
from nanocavity.full_counting import *
from nanocavity.rate_equation import *
from nanocavity.distributions import *

def jc_rates():
    [d1, d2, a], [Nf1, Nf2, Nb] = \
            composite(fermion_modes=2, boson_modes=1, max_bosons=1)
    H0 = 0.4 * Nf1 + (0.4 +  0.9) * Nf2 + Nb
    Hint = 0.3 * (a.d * d1.d * d2 + a * d2.d * d1)
    H = H0 +  Hint
    e, v = H.eigh()
    VL = np.linspace(-2, 3, 11)
    VR = 0
    #electrodes transition rates
    GpL, GmL = transition_rate(e, v, [d1, d2], 1e-3*np.eye(2), mu=VL, kT=1e-2)
    GpR, GmR = transition_rate(e, v, [d1, d2], 1e-3*np.eye(2), mu=VR, kT=1e-2)
    GL, GR = transition_rate_matrix(GpL + GmL, GpR + GmR)

    #damping matrix
    Kp, Km = transition_rate(e, v, [a], 1, kT=1e-2, bath='bosonic')
    K = Kp + Km
    return Kp, Km, GL, GpR, GmR

def test_M_matrix():
    Kp, Km, GL, GpR, GmR = jc_rates()
    E, x = M_matrix(Kp, Km, GL, GpR, GmR, 1e-3)
    index = x.size // 2
    #As the probability is conserved all eigenvalues at \chi= has to be <=0
    assert np.all(np.round(np.real(E[index, index]), 10) <= 0) 
test_M_matrix()
def test_cumulants():
    Kp, Km, GL, GpR, GmR = jc_rates()
    I_fcs_ph, I_fcs_el, _, _, _ = cumulants(Kp, Km, GL, GpR, GmR, 1e-3)
   
    vr, k, _ = GpR.shape
    GR = (GpR + GmR).reshape(1, vr, k, k)
    Gamma = (Kp + Km)[np.newaxis, np.newaxis] + GL + GR
    P = populations(Gamma)
    Iph = photo_current(Kp, Km, P)
    Iel = electro_current(GpR - GmR, P, electrode='right')
    
    assert np.allclose(I_fcs_ph, Iph)
    assert np.allclose(I_fcs_el, Iel)
test_cumulants()
