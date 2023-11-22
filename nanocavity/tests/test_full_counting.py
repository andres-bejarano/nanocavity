import numpy as np
import nanocavity.full_counting as nfcs
import nanocavity.rate_equation as nre
import nanocavity.operators as no



def jc_rates():
    H, L = no.H_tls_nc(Eg=0.4, delta=0.9, omega=1.0, coupling=0.3)
    e, v = H.eigh()
    VL = np.linspace(-2, 3, 3)
    VR = np.linspace(-3, 4, 2)
    #electrodes transition rates
    GpL, GmL = nre.transition_rate(e, v, L[:2], 1e-3*np.eye(2), mu=VL, kT=1e-2)
    GpR, GmR = nre.transition_rate(e, v, L[:2], 1e-3*np.eye(2), mu=VR, kT=1e-2)

    #damping matrix
    Kp, Km = nre.transition_rate(e, v, L[2], 1, kT=1e-2, bath='bosonic')
    return Kp, Km, GpL, GmL, GpR, GmR

def test_M_matrix():
    Kp, Km, GpL, GmL, GpR, GmR = jc_rates()
    E, x = nfcs.M_matrix(Kp, Km, GpL, GmL, GpR, GmR)
    index = x.size // 2
    #As the probability is conserved all eigenvalues at \chi= has to be <=0
    assert np.all(np.round(np.real(E[index, index]), 10) <= 0) 

#its also a test for derivates
#the real part of E is linear in a range of 1e-6
#whereas the real part is parabolic in 1e-9
def test_E_max():
    p = 1e-15
    n = 12
    Kp, Km, GpL, GmL, GpR, GmR = jc_rates()
    E, zero = nfcs.E_max(Kp, Km, GpL, GmL, GpR, GmR, p=p, ninter=n)
    E_re = np.real(E)
    E_im = np.imag(E)
    
    NphL, NphR, NphM = nfcs.d1(E_im[:, zero, :, :], zero)
    
    assert np.allclose(NphL, NphR)
    assert np.allclose(NphL, NphM)


    NelL, NelR, NelM = nfcs.d1(E_im[zero, :, :, :], zero)

    assert np.allclose(NelL, NelR)
    assert np.allclose(NelL, NelM)
    
    ZphL, ZphR, ZphM = nfcs.d2(E_re[:, zero, :, :], zero, p=p)

    assert np.allclose(ZphL, ZphR)
    assert np.allclose(ZphL, ZphM)
    
    ZelL, ZelR, ZelM = nfcs.d2(E_re[zero, :, :, :], zero)

    assert np.allclose(ZelL, ZelR)
    assert np.allclose(ZelL, ZelM)



    covL, covR, covM = nfcs.dxy(E_re, zero)

    assert np.allclose(covL, covR)
    assert np.allclose(covL, covM)


def test_cumulants():
    Kp, Km, GpL, GmL, GpR, GmR = jc_rates()
    Nph, Nel, _, _, _ = nfcs.cumulants(Kp, Km, GpL, GmL, GpR, GmR, p=1e-9)
    GL = (GpL + GmL)[:, None]  # VL, VR
    GR = (GpR + GmR)[None, :]
    Gamma = (Kp + Km)[np.newaxis, np.newaxis] + GL + GR
    P = nre.populations(Gamma)
    Iph = nre.photo_current(Kp, Km, P)
    Iel = -nre.electro_current(GpR - GmR, P, electrode='right')
    #we are counting the number of electrons that get in the right electrode
    #then we have compare with -Iel to remove the charge '-e'
    assert np.allclose(Nph, Iph)
    assert np.allclose(Nel, Iel)
