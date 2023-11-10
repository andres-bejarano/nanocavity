import numpy as np
import secondquant.composite as sc
import nanocavity.rate_equation as nre
import nanocavity.distributions as ndist

def rate_parameters(modes=1):
    d = []
    Nf = []
    d, Nf = sc.composite(fermion_modes=modes)
    if modes == 1: 
        g = 0.2
        e, v = Nf.eigsh()
        return [d], e, v, g
    else: 
        g = np.arange(modes ** 2).reshape(modes, modes)
        H = 0
        for Nfi in Nf:
            H += Nfi
        e, v = H.eigsh()
        return d, e, v, g

def test_matrix_elements():
    for i in range(1, 4):
        d, e, v, g =  rate_parameters(i)
        M = nre.matrix_elements(d, v, g)
        assert M.shape == (len(e), len(e))

def test_fermi_matrix():
    E = [0, 1, 2]
    mu = np.linspace(-1, 1, 3)
    A = nre.fermi_matrix(E=E, mu=mu) 
    DE = np.array(E).reshape(1, -1, 1) - np.array(E).reshape(1, 1, -1)
    B = ndist.fermi_dirac(DE, mu=mu.reshape(-1, 1, 1))
    assert np.allclose(A, B)

def test_bose_matrix():
    E = [0, 1, 2]
    A = nre.bose_matrix(E=E)
    DE = np.array(E).reshape(-1, 1) - np.array(E).reshape(1, -1)
    B = ndist.bose_einstein(DE)
    np.fill_diagonal(B, 0)
    assert np.allclose(A, B)

def test_transition_rate():
    for i in range(1, 4):
        d, e, v, g = rate_parameters(i)
        Gp, Gm = nre.transition_rate(e, v, d, g)
        assert np.allclose(Gp[0].diagonal(), 0)
        assert np.allclose(Gm[0].diagonal(), 0)

def test_transition_rate():
    b, Nb = sc.composite(boson_modes=1, max_bosons=2)
    e, v = Nb.eigsh()
    A = np.diag(1 + np.arange(2), 1)
    KpA = A.T * nre.bose_matrix(e)
    KmA = A * (1 + nre.bose_matrix(-e))
    Kp, Km = nre.transition_rate(e, v, [b], g=1, bath='bosonic')
    assert np.allclose(Kp, KpA)
    assert np.allclose(Km, KmA)

def test_populations():
    for i in range(1, 4):
        d, e, v, g = rate_parameters(i)
        Gp, Gm = nre.transition_rate(e, v, d, g)
        GL, GR = nre.transition_rate_matrix(Gp + Gm, Gp + Gm)
        P = nre.populations(GL + GR)
        assert np.allclose(P[0, 0].sum(), 1)

def test_asymmetrical_bias():

    #chemical potential
    VL = np.linspace(-3, 3, 3)
    VR = np.linspace(-3, 2, 4)

    #leads-exciton couplings
    gl = [[1., 0], [0, 0.8]]
    gr = [[0.4, 0], [0, 0.2]]

    #Two level system operators
    [d1, d2], [Nf1, Nf2] = sc.composite(fermion_modes=2)

    #Hamiltonian diagonalization
    delta = 1.
    H = delta * Nf2
    E, v = H.eigsh()

    #transition rates matrix
    GpL, GmL = nre.transition_rate(E, v, [d1, d2], gl, mu=VL)
    GpR, GmR = nre.transition_rate(E, v, [d1, d2], gr, mu=VR)
    GL, GR = nre.transition_rate_matrix(GpL + GmL, GpR + GmR)

    #populations
    Gamma = GL + GR
    P = nre.populations(GL + GR)
    assert np.allclose(P.sum(axis=2), 1)

def test_electro_current_single_level():
    
    #chemical potential
    VL = np.linspace(-3, 3, 3)
    VR = np.linspace(-2, 4, 5)

    #leads-exciton couplings
    gl = 0.8
    gr = 0.2

    #Two level system operators
    d, Nf = sc.composite(fermion_modes=1)

    #Hamiltonian diagonalization
    H = Nf
    E, v = H.eigsh()

    #transition rates matrix
    GpL, GmL = nre.transition_rate(E, v, [d], gl, mu=VL)
    GpR, GmR = nre.transition_rate(E, v, [d], gr, mu=VR)
    GL, GR = nre.transition_rate_matrix(GpL + GmL, GpR + GmR)

    #populations
    P = nre.populations(GL + GR)
    
    #current
    IL = nre.electro_current(GpL - GmL, P, 'left')
    IR = nre.electro_current(GpR - GmR, P, 'right')
    assert np.allclose(IL, -IR)

def test_electro_current_two_level():
    #chemical potential
    VL = np.linspace(-3, 2, 2)
    VR = np.linspace(-2, 1, 3)

    #leads-exciton couplings
    gl = [[1, 0], [0, 0.8]]
    gr = [[0.5, 0], [0, 0.3]]

    #Two level system operators
    [d1, d2], [Nf1, Nf2] = sc.composite(fermion_modes=2)

    #Hamiltonian diagonalization
    delta = 1.
    H = delta * Nf2
    E, v = H.eigsh()

    #transition rates matrix
    GpL, GmL = nre.transition_rate(E, v, [d1, d2], gl, mu=VL)
    GpR, GmR = nre.transition_rate(E, v, [d1, d2], gr, mu=VR)
    GL, GR = nre.transition_rate_matrix(GpL + GmL, GpR + GmR)

    #populations
    P = nre.populations(GL + GR)

    #current
    IL = nre.electro_current(GpL - GmL, P, 'left')
    IR = nre.electro_current(GpR - GmR, P, 'right')
    
    assert np.allclose(IL, -IR)

def test_power_spectrum():
    #coupling with leads
    gl = 1e-3 * np.eye(2)
    gr = 1e-3 * np.eye(2)

    g = gl[0, 0]

    #light-matter coupling
    eg = 0.2
    delta = 0.9
    omega_c = 1
    lambd = 0.3
    theta = 0.5 * np.angle((omega_c - delta) +  lambd*2j)

    #damping
    kappa = 1


    #VL,VR, kBT
    VL = np.linspace(-3, 3, 5)
    VR = np.linspace(-3, 3, 2)
    kT = 1e-2

    #hamiltonian
    [d1, d2, a], [Nf1, Nf2, Nb] = sc.composite(fermion_modes=2, boson_modes=1, max_bosons=1)
    H0 = eg * Nf1 + (eg +  delta) * Nf2 + omega_c * Nb
    Hint = a.d * d1.d * d2 + a * d2.d * d1

    #Hamiltonian diagonalization
    H = H0 + lambd * Hint
    e, v = H.eigsh()
    idx = [0, 1, 2, 5]
    e = e[idx]
    v = v[:, idx]

    #electrodes transition rates
    GpL, GmL = nre.transition_rate(e, v, [d1, d2], gl, mu=VL, kT=kT)
    GpR, GmR = nre.transition_rate(e, v, [d1, d2], gl, mu=VR, kT=kT)
    GL, GR = nre.transition_rate_matrix(GpL + GmL, GpR + GmR)

    #damping matrix
    Kp, Km = nre.transition_rate(e, v, [a], kappa, kT=kT, bath='bosonic')
    K = Kp + Km
    #transtion rates matrix
    Gamma = K[np.newaxis, np.newaxis] + GL + GR

    #populations
    P = nre.populations(Gamma)
    Pm = P[:, :, 2][np.newaxis]
    Pp = P[:, :, 3][np.newaxis]

    #photo-current numerical
    omega = np.linspace(0, 3, 3)
    Egm = e[2] - e[1]
    Egp = e[3] - e[1]
    wide_gm = K[1, 2]
    wide_gp = K[1, 3]
    Lm = ndist.lorentzian(Egm - omega, wide_gm)[:, np.newaxis, np.newaxis]
    Lp = ndist.lorentzian(Egp - omega, wide_gp)[:, np.newaxis, np.newaxis]

    Ig = np.sin(theta) ** 2 * Pm * Lm + np.cos(theta) ** 2 * Pp * Lp
    I = nre.power_spectrum(Kp, Km, P, e, omega)

    assert np.allclose(Ig, I)

def test_photo_current():
    #parameters
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

    [d1, d2, a], [Nf1, Nf2, Nb] = \
        sc.composite(fermion_modes=2, boson_modes=1, max_bosons=1)
    H0 = Eg * Nf1 + (Eg +  delta) * Nf2 + omegac * Nb
    Hint = coupling * (a.d * d1.d * d2 + a * d2.d * d1)
    H = H0 +  Hint
    e, v = H.eigh()
    idx = [0, 1, 3, 4, 5]
    e = e[idx]
    v = v[:, idx]

    #electrodes transition rates
    GpL, GmL = nre.transition_rate(e, v, [d1, d2], g, mu=VL, kT=kT)
    GpR, GmR = nre.transition_rate(e, v, [d1, d2], g, mu=VR, kT=kT)

    GL, GR = nre.transition_rate_matrix(GpL + GmL, GpR + GmR)

    #damping matrix
    Kp, Km = nre.transition_rate(e, v, [a], kappa, kT=kT, bath='bosonic')
    K = Kp + Km

    #transtion rates matrix
    Gamma = K[np.newaxis, np.newaxis] + GL + GR
    P = nre.populations(Gamma)

    I = np.round(nre.photo_current(Kp, Km, P), 4)
    print(I)
    assert np.allclose(I, g[0, 0]/2)

def test_bath_system_bath_rate():
    #parameters

    omegac = 1
    kappa = 1
    kT = 1e-1
    VL = np.linspace(-105, 5, 101)
    VR = np.linspace(-3, 103, 101)

    m = 2
    n = np.arange(11)

    #hamiltonian
    a, Nb = sc.composite(boson_modes=1, max_bosons=max(n))
    e, v = Nb.eigh()

    #transtion rates, populations and spectrum
    Kp, Km = nre.transition_rate(e, v, [a], kappa, kT=kT, bath='bosonic')
    K = Kp + Km
    M = nre.bath_system_bath_rate(e, v, [a + a.d], m, VL=VL, VR=VR, kT=kT)
    P = nre.populations(K[np.newaxis, np.newaxis]  + M)


    #analytics
    E = abs(np.array(omegac).reshape(1, 1, -1))
    bias = abs(VL.reshape(-1, 1, 1) - VR.reshape(1, -1, 1))
    Gup = kappa * ndist.bose_einstein(omegac, kT=kT) + \
            m *  (ndist.Fermi_cb(bias-E, kT) + ndist.Fermi_cb(-bias-E, kT=kT))
    Gdw = kappa * (1 + ndist.bose_einstein(omegac, kT=kT)) + \
            m * (ndist.Fermi_cb(bias+E, kT) + ndist.Fermi_cb(-bias+E, kT=kT))
    x= Gup / Gdw
    gamma = (1 - x)/(1 - x ** (max(n) + 1))
    Pn = gamma * x ** n

    assert np.allclose(P, Pn)
