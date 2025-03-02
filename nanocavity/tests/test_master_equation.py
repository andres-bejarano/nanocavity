import numpy as np
import nanocavity.master_equation as nme
import nanocavity.operators as no
from scipy.linalg import eig
import secondquant as sq
import pytest
import nanocavity.distributions as ndist


def test_reduced_population():
    _, [nf, nb] = sq.composite(fermion_modes=1, boson_modes=[3])
    rho = np.random.rand(*nf.shape)
    pops = np.diag(rho)
    # Checking for higher dimensional (bias maps) density matrices
    rho = np.random.rand(3, 2, 8, 8)
    f_pop0 = np.sum(np.diag(rho[1, 1, 0:4, 0:4]))
    f_pop1 = np.sum(np.diag(rho[2, 0, 4:, 4:]))
    fermi_red3 = nme.reduced_population(nf, rho)
    assert np.allclose(fermi_red3[1, 1, 0], f_pop0)
    assert np.allclose(fermi_red3[2, 0, 1], f_pop1)

    _, [nf1, nf2, nb1, nb2] = sq.composite(fermion_modes=2, boson_modes=[3, 2])
    rho = np.random.rand(*nf1.shape)
    pops = np.diag(rho)
    p_f1 = nme.reduced_population(nf1, rho)
    p_f1N = nme.reduced_population(nf1, rho[None, None])
    assert np.allclose(p_f1, p_f1N)
    p_f2 = nme.reduced_population(nf2, rho)
    p_f = nme.reduced_population(nf1 + nf2, rho)
    p_b1 = nme.reduced_population(nb1, rho)
    p_b2 = nme.reduced_population(nb2, rho)
    for n in range(len(p_b1)):
        idx = nb1.where(n)
        assert np.allclose(np.sum(pops[idx]), p_b1[n])
    for n in range(len(p_b2)):
        idx = nb2.where(n)
        assert np.allclose(np.sum(pops[idx]), p_b2[n])
    for n in range(len(p_f1)):
        id1 = nf1.where(n)
        id2 = nf2.where(n)
        assert np.allclose(np.sum(pops[id1]), p_f1[n])
        assert np.allclose(np.sum(pops[id2]), p_f2[n])
    for n in range(len(p_f)):
        idx = (nf1 + nf2).where(n)
        assert np.allclose(np.sum(pops[idx]), p_f[n])

    nr_list = [nf1, nf2, nb1, nb2]
    P_102 = nme.reduced_population(nr_list[0:3], rho, [1, 0, 2])
    P_1020 = nme.reduced_population(nr_list, rho, [1, 0, 2, 0])
    P_1021 = nme.reduced_population(nr_list, rho, [1, 0, 2, 1])
    P_1022 = nme.reduced_population(nr_list, rho, [1, 0, 2, 2])
    assert np.allclose(P_1020 + P_1021 + P_1022, P_102)


def test_eig_norm():
    d = 25
    # build symmetric, nonhermitian matrix
    L = np.random.rand(d, d) + 1j * np.random.rand(d, d)
    L += L.T
    E, vl, vr = eig(L, left=True)

    E1, wr = np.linalg.eig(L)
    E2, wl = np.linalg.eig(L.conj().T)

    assert np.allclose(E, E1)
    assert np.allclose(E1, E2.conj())

    assert np.allclose(vl, wl)
    assert np.allclose(vr, wr)

    # now check our orthonormalization routine
    E, vl, vr = nme.eig_norm(L)

    # orthonormalization
    norm = np.einsum("ai,aj->ij", vl.conj(), vr)
    assert np.allclose(norm, np.eye(d))

    # recovering eigenvalues
    E1 = np.einsum("ai,ab,bi->i", vl.conj(), L, vr)
    assert np.allclose(E, E1)

def test_eig_norm_nonsymmetric():
    d = 25
    # build nonsymmetric, nonhermitian matrix
    L = np.random.rand(d, d) + 1j * np.random.rand(d, d)
    E, vl, vr = nme.eig_norm(L)
    # orthonormalization
    norm = np.einsum("ai,aj->ij", vl.conj(), vr)
    assert np.allclose(norm, np.eye(d))
    # recovering eigenvalues
    E1 = np.einsum("ai,ab,bi->i", vl.conj(), L, vr)
    assert np.allclose(E, E1)


@pytest.fixture(scope="module", params=[0.01, 1.0])
def kT(request):
    return request.param


@pytest.fixture(scope="module", params=[2, 5])
def bosons(request):
    return request.param


@pytest.fixture(scope="module", params=[0.01, 0.001])
def rate(request):
    return request.param


def test_liouvillian(kT, bosons, rate):
    [c, a], [Nf, Nb] = sq.composite(fermion_modes=1, boson_modes=1, max_bosons=bosons)
    H = Nf + 0.1 * Nb + 0.01 * (c.d * a + a.d * c)
    basis = H.eigh()
    c_ops = no.collapses(c, basis, kT, "fermionic", rate)
    a_ops = no.collapses(a, basis, kT, "bosonic", rate)
    for ops in (c_ops, a_ops, c_ops + a_ops):
        L = no.liouvillian(H, ops)
        # check determinant of L is zero
        assert np.isclose(np.linalg.det(L), 0)

@pytest.fixture(scope="module", params=["eig", "solve"])
def method(request):
    return request.param


@pytest.fixture(scope="module", params=range(3))
def row(request):
    return request.param


@pytest.fixture(scope="module", params=[1e-12, 1e-200])
def scale(request):
    return request.param


def test_stationary(kT, bosons, rate, method):
    [c, a], [Nf, Nb] = sq.composite(fermion_modes=1, boson_modes=1, max_bosons=bosons)
    H = Nf + 0.1 * Nb + 0.01 * (c.d * a + a.d * c)
    basis = H.eigh()
    c_ops = no.collapses(c, basis, kT, "fermionic", rate)
    a_ops = no.collapses(a, basis, kT, "bosonic", rate)
    L = no.liouvillian(H, c_ops + a_ops)
    rho = nme.stationary(L, method=method, check=True, tol=-1e8)


def test_stationary_solve(row, scale):
    [c, a], [Nf, Nb] = sq.composite(fermion_modes=1, boson_modes=1, max_bosons=3)
    H = Nf + 0.1 * Nb + 0.01 * (c.d * a + a.d * c)
    rate = 0.1
    kT = 0.01
    basis = H.eigh()
    c_ops = no.collapses(c, basis, kT, "fermionic", rate)
    a_ops = no.collapses(a, basis, kT, "bosonic", rate)
    L = no.liouvillian(H, c_ops + a_ops)
    rho = nme.stationary(L, method="solve", row=row, scale=scale, check=True, tol=-1e8)


def test_stationary_single_cavity_mode():
    # parameters
    hw_ph = 1
    kappa = 0.1
    kT = 1e-1
    n = np.arange(10)

    a, Nb = sq.composite(boson_modes=1, max_bosons=max(n))

    H = hw_ph * Nb
    basis = H.eigh()
    # numerics
    cp, cm = no.collapses(a, basis, kT, "bosonic", kappa, total=False)
    L = no.liouvillian(H, cp + cm)
    rho_st = nme.stationary(L)
    Pn = rho_st.diagonal()

    # photons coming out
    In = nme.average(no.jump(cm) - no.jump(cp), rho_st)
    n_average = nme.average(no.jump(cm), rho_st)

    # analytics
    Gup = kappa * ndist.bose_einstein(hw_ph, kT)
    Gdw = kappa * (1 + ndist.bose_einstein(hw_ph, kT))
    x = Gup / Gdw
    gamma = (1 - x) / (1 - x ** (max(n) + 1))
    Pa = gamma * x**n

    # tr(Jrho) = tr(a rho a^\dagger) = <n>
    n_average_a = kappa * ndist.bose_einstein(hw_ph, kT)

    assert np.allclose(Pn, Pa)
    assert np.allclose(In, 0)
    assert np.allclose(n_average, n_average_a)


def test_spectrum():
    [c, a], [Nf, Nb] = sq.composite(fermion_modes=1, boson_modes=1, max_bosons=2)
    H = Nf + 0.1 * Nb + 0.01 * (c.d * a + a.d * c)
    kT = 0.1
    rate = 1e-3
    basis = H.eigh()
    c_ops = no.collapses(c, basis, kT, "fermionic", rate)
    a_ops = no.collapses(a, basis, kT, "bosonic", rate)
    L = no.liouvillian(H, c_ops + a_ops)
    frequency = np.arange(1, 10)
    I = nme.spectrum(L, a, frequency, verbose=False)
    I2, Mk, E = nme.spectrum(L, a, frequency, verbose=False, ret_data=True)
    assert np.allclose(I, I2)
    assert np.all(Mk >= -1e-30)  # tolerate tiny negative values?
    assert np.all(Mk[1:] <= Mk[:-1])  # entries sorted descending?
    I3 = nme.spectrum(L, a, frequency, cutoff=10)
    assert np.allclose(I3, 0)

    rho_st = nme.stationary(L, method="solve")
    I4 = nme.spectrum(L, a, frequency, rho_st=rho_st)
    assert np.allclose(I, I4)

    # check also with a number instead of array
    I5 = nme.spectrum(L, a, frequency[0], verbose=False)
    assert np.isclose(I[0], I5)

    # test list
    I6 = nme.spectrum(L, a, [1, 2], verbose=False)
    assert np.allclose(I[:2], I6)


def test_g2():
    # just a single harmonic oscillator, H = a.d * a, coupled to bath
    a, H = sq.composite(fermion_modes=0, boson_modes=1, max_bosons=6)
    basis = H.eigh()
    ap, am = no.collapses(a, basis, 0.1, "bosonic", 0.1, total=False)
    L = no.liouvillian(H, ap + am)
    time_delay = np.linspace(0, 200, 2)
    for method in ["eigen", "direct"]:
        g2, G1, M, E = nme.g2(L, a, time_delay, method, ret_data=True)
        assert np.isclose(g2[0], 2)  # g2(0) = 2 in thermal equilibrium
        assert np.isclose(g2[-1], 1)  # g2(infty) = 1, uncorrelated

    time_delay = np.arange(200)
    g2 = nme.g2(L, a, time_delay)

    assert np.all(g2[:-1] > g2[1:])  # decreasing function of delay
    g2b, G1, Mk, E = nme.g2(L, a, time_delay, ret_data=True)
    assert np.allclose(g2, g2b)

    rho_st = nme.stationary(L)
    g2c = nme.g2(L, a, time_delay, rho_st=rho_st)
    assert np.allclose(g2, g2c)

    # check also with a number instead of array
    g2d = nme.g2(L, a, time_delay[0], verbose=False)
    assert np.isclose(g2[0], g2d)

    # check list
    g2e = nme.g2(L, a, [0, 1], verbose=False)
    assert np.allclose(g2[:2], g2e)

    # analytic same-time
    g2_zero = nme.g2_zero(a, rho_st)
    assert np.isclose(2, g2_zero)
