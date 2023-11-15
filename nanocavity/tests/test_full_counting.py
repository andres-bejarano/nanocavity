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
    VL = np.linspace(-3, 3, 101)
    VR = 0
    #electrodes transition rates
    GpL, GmL = transition_rate(e, v, [d1, d2], 1e-3*np.eye(2), mu=VL, kT=1e-2)
    GpR, GmR = transition_rate(e, v, [d1, d2], 1e-3*np.eye(2), mu=VR, kT=1e-2)

    #damping matrix
    Kp, Km = transition_rate(e, v, [a], 1, kT=1e-2, bath='bosonic')
    K = Kp + Km
    return Kp, Km, GpL, GmL, GpR, GmR

def test_M_matrix():
    Kp, Km, GpL, GmL, GpR, GmR = jc_rates()
    E, x = M_matrix(Kp, Km, GpL, GmL, GpR, GmR)
    index = x.size // 2
    #As the probability is conserved all eigenvalues at \chi= has to be <=0
    assert np.all(np.round(np.real(E[index, index]), 10) <= 0) 

#its also a test for derivates
#the real part of E is linear in a range of 1e-6
#whereas the real part is parabolic in 1e-9
def test_E_max():
    Kp, Km, GpL, GmL, GpR, GmR = jc_rates()
    E, zero = E_max(Kp, Km, GpL, GmL, GpR, GmR)
    E_re = np.real(E)
    E_im = np.imag(E)
    
    NphL, NphR, NphM = d1(E_im[:, zero, :, :], zero)
    
    assert np.allclose(NphL, NphR)
    assert np.allclose(NphL, NphM)


    NelL, NelR, NelM = d1(E_im[zero, :, :, :], zero)

    assert np.allclose(NelL, NelR)
    assert np.allclose(NelL, NelM)
    
    ZphL, ZphR, ZphM = d2(E_re[:, zero, :, :], zero)

    assert np.allclose(ZphL, ZphR)
    assert np.allclose(ZphL, ZphM)

    ZelL, ZelR, ZelM = d2(E_re[zero, :, :, :], zero)

    assert np.allclose(ZelL, ZelR)
    assert np.allclose(ZelL, ZelM)



    covL, covR, covM = dxy(E_re, zero)

    assert np.allclose(covL, covR)
    assert np.allclose(covL, covM)


def test_cumulants():
    Kp, Km, GpL, GmL, GpR, GmR = jc_rates()
    Nph, Nel, _, _, _ = cumulants(Kp, Km, GpL, GmL, GpR, GmR, 1e-3)
  
    GL, GR = transition_rate_matrix(GpL + GmL, GpR + GmR)
    Gamma = (Kp + Km)[np.newaxis, np.newaxis] + GL + GR
    P = populations(Gamma)
    Iph = photo_current(Kp, Km, P)
    Iel = electro_current(GpR - GmR, P, electrode='right')
    #we are counting the number of electrons that get in the right electrode
    #then we have compare with -Iel to remove the charge '-e'
    assert np.allclose(Nph, Iph)
    assert np.allclose(Nel, -Iel)
