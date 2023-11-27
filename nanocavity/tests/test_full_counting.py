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



def test_cumulants():
    Kp, Km, GpL, GmL, GpR, GmR = jc_rates()
    Iphfcs, Ielfcs, _, _, _ = nfcs.cumulants(Kp, Km, GpL, GmL, GpR, GmR, p=1e-9)
    GL = (GpL + GmL)[:, None]  # VL, VR
    GR = (GpR + GmR)[None, :]
    Gamma = (Kp + Km)[np.newaxis, np.newaxis] + GL + GR
    P = nre.populations(Gamma)
    Iphnc = nre.photo_current(Kp, Km, P)
    Ielnc = nre.electro_current(GpR - GmR, P, electrode='right')
    assert np.allclose(Iphfcs, Iphnc)
    assert np.allclose(Ielfcs, Ielnc)
