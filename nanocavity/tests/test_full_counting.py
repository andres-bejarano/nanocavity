import numpy as np
import nanocavity.full_counting as nfcs
import nanocavity.rate_equation as nre
import nanocavity.operators as no
import nanocavity.tls as tls


def jc_rates():
    H0, Hint, L = tls.Hamiltonian(
        "nanocavity", Eg=0.4, delta=0.9, omegac=1.0, coupling=0.3
    )
    H = H0 + Hint
    e, v = H.eigh()
    VL = np.linspace(-2, 3, 3)
    VR = np.linspace(-3, 4, 2)
    # electrodes transition rates
    GpL, GmL = nre.transition_rate(e, v, L[:2], 1e-3 * np.eye(2), mu=VL, kT=1e-2)
    GpR, GmR = nre.transition_rate(e, v, L[:2], 1e-3 * np.eye(2), mu=VR, kT=1e-2)

    # damping matrix
    Kp, Km = nre.transition_rate(e, v, L[2], 1, kT=1e-2, bath="bosonic")
    return Kp, Km, GpL, GmL, GpR, GmR


def test_transition_rates_fourier():
    Kp, Km, GpL, GmL, GpR, GmR = jc_rates()
    E, x = nfcs.transition_rates_fourier(Kp, Km, GpL, GmL, GpR, GmR)
    index = x.size // 2
    # As the probability is conserved all eigenvalues at \chi= has to be <=0
    assert np.all(np.round(np.real(E[index, index]), 10) <= 0)


def test_E_max():
    Kp, Km, GpL, GmL, GpR, GmR = jc_rates()
    E, x = nfcs.transition_rates_fourier(Kp, Km, GpL, GmL, GpR, GmR)
    Emax, zero, x = nfcs.E_max(Kp, Km, GpL, GmL, GpR, GmR)

    # E[x, y, vl, vr, dim(H)]
    Nx = E.shape[0]
    Nvl = E.shape[2]
    Nvr = E.shape[3]
    for i, j, k, l in zip(range(Nx), range(Nx), range(Nvl), range(Nvr)):

        assert Emax[i, j, k, l] == max(E[i, j, k, l])


def test_cumulants():
    Kp, Km, GpL, GmL, GpR, GmR = jc_rates()
    Iphfcs, Ielfcs, _, _, _ = nfcs.cumulants(Kp, Km, GpL, GmL, GpR, GmR, p=1e-9)
    GL = (GpL + GmL)[:, None]  # VL, VR
    GR = (GpR + GmR)[None, :]
    Gamma = (Kp + Km)[np.newaxis, np.newaxis] + GL + GR
    P = nre.populations(Gamma)
    Iphnc = nre.photo_current(Kp, Km, P)
    Ielnc = nre.electro_current(GpR - GmR, P, electrode=1)
    assert np.allclose(Iphfcs, Iphnc)
    assert np.allclose(Ielfcs, Ielnc)
