import numpy as np
import secondquant as sq
import nanocavity.rate_equation as nre
import nanocavity.distributions as ndist


def rate_parameters(modes=1):
    d = []
    Nf = []
    d, Nf = sq.composite(fermion_modes=modes)
    if modes == 1:
        g = 0.2
        e, v = Nf.eigsh()
        return [d], e, v, g
    else:
        g = np.arange(modes**2).reshape(modes, modes)
        H = 0
        for Nfi in Nf:
            H += Nfi
        e, v = H.eigsh()
        return d, e, v, g


def peak(height, position, fwhm, wlist):
    return height * ndist.lorentzian(wlist - position, fwhm)


def parameters(coupling, omegac, delta, kappa, gt, gs):
    detuning = omegac - delta
    theta = 0.5 * np.arctan(2 * coupling / detuning)

    Kpg = kappa * np.cos(theta) ** 2
    Wpg = 2 * (gt[0, 0] + gs[0, 0]) + Kpg

    Kmg = kappa * np.sin(theta) ** 2
    Wmg = 2 * (gt[0, 0] + gs[0, 0]) + Kmg

    Kme = kappa * (1 + np.sin(theta) ** 2)
    Wme = 2 * (gt[0, 0] + gs[0, 0]) + Kme

    Kpe = kappa * (1 + np.cos(theta) ** 2)
    Wpe = 2 * (gt[0, 0] + gs[0, 0]) + Kpe

    return Kpg, Wpg, Kmg, Wmg, Wme, Wpe


def test_matrix_elements():
    for i in range(1, 4):
        d, e, v, g = rate_parameters(i)
        M = nre.matrix_elements(d, v, g)
        assert M.shape == (len(e), len(e))


def test_fermi_matrix():
    E = [0, 1, 2]
    kT = 0.1
    mu = np.linspace(-1, 1, 3)
    A = nre.fermi_matrix(E, kT, mu)
    DE = np.array(E).reshape(1, -1, 1) - np.array(E).reshape(1, 1, -1)
    B = ndist.fermi_dirac(DE, kT, mu.reshape(-1, 1, 1))
    assert np.allclose(A, B)


def test_bose_matrix():
    E = [0, 1, 2]
    kT = 0.1
    A = nre.bose_matrix(E, kT)
    DE = np.array(E).reshape(-1, 1) - np.array(E).reshape(1, -1)
    B = ndist.bose_einstein(DE, kT)
    np.fill_diagonal(B, 0)
    assert np.allclose(A, B)


def test_transition_rate():
    for i in range(1, 4):
        d, e, v, g = rate_parameters(i)
        Gp, Gm = nre.transition_rate(e, v, d, g, 0.1)
        assert np.allclose(Gp[0].diagonal(), 0)
        assert np.allclose(Gm[0].diagonal(), 0)


def test_transition_rate():
    b, Nb = sq.composite(boson_modes=1, max_bosons=2)
    e, v = Nb.eigsh()
    A = np.diag(1 + np.arange(2), 1)
    kT = 0.1
    KpA = A.T * nre.bose_matrix(e, kT)
    KmA = A * (1 + nre.bose_matrix(-e, kT))
    Kp, Km = nre.transition_rate(e, v, [b], 1, kT, bath="bosonic")
    assert np.allclose(Kp, KpA)
    assert np.allclose(Km, KmA)


def test_populations():
    for i in range(1, 4):
        d, e, v, g = rate_parameters(i)
        Gp, Gm = nre.transition_rate(e, v, d, g, 0.1)
        G = Gp + Gm
        GL = G[:, None]  # VL, VR
        GR = G[None, :]
        P = nre.populations(GL + GR)
        assert np.allclose(P[0, 0].sum(), 1)


def test_asymmetrical_bias():

    # chemical potential
    VL = np.linspace(-3, 3, 3)
    VR = np.linspace(-3, 2, 4)

    # leads-exciton couplings
    gl = [[1.0, 0], [0, 0.8]]
    gr = [[0.4, 0], [0, 0.2]]

    # Two level system operators
    [d1, d2], [Nf1, Nf2] = sq.composite(fermion_modes=2)

    # Hamiltonian diagonalization
    delta = 1.0
    H = delta * Nf2
    E, v = H.eigsh()

    # transition rates
    kT = 0.1
    GpL, GmL = nre.transition_rate(E, v, [d1, d2], gl, kT, mu=VL)
    GpR, GmR = nre.transition_rate(E, v, [d1, d2], gr, kT, mu=VR)
    GL = (GpL + GmL)[:, None]  # VL, VR
    GR = (GpR + GmR)[None, :]

    # populations
    Gamma = GL + GR
    P = nre.populations(GL + GR)
    assert np.allclose(P.sum(axis=2), 1)


def test_electro_current_single_level():
    # chemical potential
    VL = np.linspace(-3, 3, 3)
    VR = np.linspace(-2, 4, 5)

    # leads-exciton couplings
    gl = 0.8
    gr = 0.2

    # Two level system operators
    d, Nf = sq.composite(fermion_modes=1)

    # Hamiltonian diagonalization
    H = Nf
    E, v = H.eigsh()

    # transition rates matrix
    kT = 0.1
    GpL, GmL = nre.transition_rate(E, v, d, gl, kT, mu=VL)
    GpR, GmR = nre.transition_rate(E, v, d, gr, kT, mu=VR)
    GL = (GpL + GmL)[:, None]  # VL, VR
    GR = (GpR + GmR)[None, :]

    # populations
    P = nre.populations(GL + GR)

    # current
    IL = nre.electro_current(GpL - GmL, P, 0)
    IR = nre.electro_current(GpR - GmR, P, 1)
    IL2 = nre.electro_current(GpL - GmL, P, "left")
    IR2 = nre.electro_current(GpR - GmR, P, "right")
    assert np.allclose(IL, -IR2)
    assert np.allclose(IL2, -IR)


def test_electro_current_two_level():
    # chemical potential
    VL = np.linspace(-3, 2, 2)
    VR = np.linspace(-2, 1, 3)

    # leads-exciton couplings
    gl = [[1, 0], [0, 0.8]]
    gr = [[0.5, 0], [0, 0.3]]

    # Two level system operators
    [d1, d2], [Nf1, Nf2] = sq.composite(fermion_modes=2)

    # Hamiltonian diagonalization
    delta = 1.0
    H = delta * Nf2
    E, v = H.eigsh()

    # transition rates
    kT = 0.1
    GpL, GmL = nre.transition_rate(E, v, [d1, d2], gl, kT, mu=VL)
    GpR, GmR = nre.transition_rate(E, v, [d1, d2], gr, kT, mu=VR)
    GL = (GpL + GmL)[:, None]  # VL, VR
    GR = (GpR + GmR)[None, :]

    # populations
    P = nre.populations(GL + GR)

    # current
    IL = nre.electro_current(GpL - GmL, P, 0)
    IR = nre.electro_current(GpR - GmR, P, 1)
    IL2 = nre.electro_current(GpL - GmL, P, "left")
    IR2 = nre.electro_current(GpR - GmR, P, "right")

    assert np.allclose(IL, -IR2)
    assert np.allclose(IL2, -IR)


def test_photo_current():
    # parameters
    omegac = 1
    delta = 0.9
    coupling = 0.3
    g = 1e-3 * np.eye(2)
    kappa = 1
    kT = 1e-2
    VL = 3
    VR = -3
    Eg = 0.4
    omega = np.linspace(-20, 20, 1001)

    [d1, d2, a], [Nf1, Nf2, Nb] = sq.composite(
        fermion_modes=2, boson_modes=1, max_bosons=1
    )
    H0 = Eg * Nf1 + (Eg + delta) * Nf2 + omegac * Nb
    Hint = coupling * (a.d * d1.d * d2 + a * d2.d * d1)
    H = H0 + Hint
    e, v = H.eigh()
    idx = [0, 1, 3, 4, 5]
    e = e[idx]
    v = v[:, idx]

    # electrodes transition rates
    GpL, GmL = nre.transition_rate(e, v, [d1, d2], g, kT, mu=VL)
    GpR, GmR = nre.transition_rate(e, v, [d1, d2], g, kT, mu=VR)
    GL = (GpL + GmL)[:, None]  # VL, VR
    GR = (GpR + GmR)[None, :]

    # damping matrix
    Kp, Km = nre.transition_rate(e, v, [a], kappa, kT=kT, bath="bosonic")
    K = Kp + Km

    # transtion rates
    Gamma = K[np.newaxis, np.newaxis] + GL + GR
    P = nre.populations(Gamma)

    I = np.round(nre.photo_current(Kp, Km, P), 4)
    assert np.allclose(I, g[0, 0] / 2)


def test_spectrum():
    # system parameters
    Eg = -0.3
    delta = 0.9
    omegac = 1
    coupling = 0.3

    # bath parameters
    gt = 1e-3 * np.eye(2)
    gs = 2e-3 * np.eye(2)
    kappa = 0.1
    kT = 1e-2
    wlist = np.linspace(0.0, 1.8, 103)

    Vt = 3
    Vs = -3

    [d1, d2, a], [Nf1, Nf2, Nb] = sq.composite(
        fermion_modes=2, boson_modes=1, max_bosons=1
    )
    H0 = Eg * Nf1 + (Eg + delta) * Nf2 + omegac * Nb
    Hint = coupling * (a.d * d1.d * d2 + a * d2.d * d1)
    H = H0 + Hint
    Een, Est = H.eigsh(k=H.shape[0])

    E0 = 0
    E01 = omegac
    Eg1 = Eg + omegac
    Ee = Eg + delta
    Ee1 = Ee + omegac
    Ege = 2 * Eg + delta
    Ege1 = Ege + omegac

    rabi = np.sqrt((omegac - delta) ** 2 + (2 * coupling) ** 2)
    Eplus = Eg + (omegac + delta) / 2 + rabi / 2
    Eminus = Eg + (omegac + delta) / 2 - rabi / 2

    E = np.array([E0, E01, Eg, Eminus, Eplus, Ee1, Ege, Ege1])
    L = []
    diff = np.abs(E[:, np.newaxis] - Een)
    L = np.argmin(diff, axis=1)
    [i0, i1, ig, im, ip, ie1, ige, ige1] = L

    GpL, GmL = nre.transition_rate(Een, Est, [d1, d2], gt, kT, mu=Vt)
    GpR, GmR = nre.transition_rate(Een, Est, [d1, d2], gs, kT, mu=Vs)
    GL = (GpL + GmL)[:, None]  # VL, VR
    GR = (GpR + GmR)[None, :]

    # damping matrix
    Kp, Km = nre.transition_rate(Een, Est, [a], kappa, kT, bath="bosonic")
    K = Kp + Km

    # transtion rates
    Gamma = K[np.newaxis, np.newaxis] + GL + GR
    P = nre.populations(Gamma)

    Kpg, Wpg, Kmg, Wmg, Wme, Wpe = parameters(coupling, omegac, delta, kappa, gt, gs)
    I = peak(Kmg * (P[0, 0, im]), Een[im] - Een[ig], Wmg, wlist)
    I += peak(Kmg * (P[0, 0, ie1]), Een[im] - Een[ig], Wpe, wlist)
    I += peak(Kpg * (P[0, 0, ip]), Een[ip] - Een[ig], Wpg, wlist)
    I += peak(Kpg * (P[0, 0, ie1]), Een[ip] - Een[ig], Wme, wlist)
    I += peak(kappa * P[0, 0, i1], omegac, 4 * gt[0, 0] + kappa, wlist)
    I += peak(kappa * P[0, 0, ige1], omegac, 4 * gs[0, 0] + kappa, wlist)

    Ire = nre.spectrum(Km, Kp, GpL + GpR, GmL + GmR, P, Een, wlist)
    assert np.allclose(I, np.squeeze(Ire))


def test_bath_system_bath_rate():
    # parameters
    omegac = 1
    kappa = 1
    kT = 1e-1
    VL = np.linspace(-105, 5, 11)
    VR = np.linspace(-3, 103, 21)

    m = 2
    n = np.arange(11)

    # hamiltonian
    a, Nb = sq.composite(boson_modes=1, max_bosons=max(n))
    e, v = Nb.eigh()

    # transtion rates, populations and spectrum
    Kp, Km = nre.transition_rate(e, v, a, kappa, kT, bath="bosonic")
    K = Kp + Km
    Mp, Mm = nre.bath_system_bath_rate(e, v, a, m, VL=VL, VR=VR, kT=kT)
    M = Mp + Mm
    P = nre.populations(K[np.newaxis, np.newaxis] + M)

    # analytics
    E = abs(np.array(omegac).reshape(1, 1, -1))
    bias = abs(VL.reshape(-1, 1, 1) - VR.reshape(1, -1, 1))
    Gup = kappa * ndist.bose_einstein(omegac, kT) + m * (
        ndist.Fermi_cb(bias - E, kT) + ndist.Fermi_cb(-bias - E, kT)
    )
    Gdw = kappa * (1 + ndist.bose_einstein(omegac, kT)) + m * (
        ndist.Fermi_cb(bias + E, kT) + ndist.Fermi_cb(-bias + E, kT)
    )
    x = Gup / Gdw
    gamma = (1 - x) / (1 - x ** (max(n) + 1))
    Pn = gamma * x**n
    assert np.allclose(P, Pn)
