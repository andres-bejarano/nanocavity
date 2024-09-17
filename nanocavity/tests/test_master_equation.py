import numpy as np
import nanocavity.master_equation as nme
import nanocavity.operators as no
from scipy.linalg import eig
import secondquant as sq
import pytest
import nanocavity.distributions as ndist

A = 0.4
B = 0.9
C = 1
D = 0.5
F = 3

def test_eig_norm():
    a = 1j * A - B
    b = 1j * C  - D
    c = 1j *  F 
    L = np.array([[a, c], [c,  b]])

    E, vl, vr = eig(L, left=True)

    E1, wr = np.linalg.eig(L)
    E2, wl = np.linalg.eig(L.conj().T)

    assert np.allclose(E, E1)
    assert np.allclose(E1, E2.conj())
    
    assert np.allclose(vl, wl)
    assert np.allclose(vr, wr)


    E, vl, vr = nme.eig_norm(L)

    norm = np.einsum("ai,ai->i", vl.conj(), vr) 
    
    assert np.allclose(norm.all(), 1)


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
    c_ops = no.collapses(c, H, kT, "fermionic", rate)
    a_ops = no.collapses(a, H, kT, "bosonic", rate)
    for ops in (c_ops, a_ops, c_ops + a_ops):
        L = no.liouvillian(H, ops)
        # check determinant of L is zero
        assert np.isclose(np.linalg.det(L), 0)
    return L

@pytest.fixture(scope="module", params=["eig", "solve"])
def method(request):
    return request.param

@pytest.fixture(scope="module", params=range(3))
def row(request):
    return request.param

@pytest.fixture(scope="module", params=[1e-10, 1.])
def scale(request):
    return request.param

def test_stationary(kT, bosons, rate, method):
    L = test_liouvillian(kT, bosons, rate)
    rho = nme.stationary(L, method=method)
    # check norm
    tr = np.trace(rho)
    assert np.isclose(tr, 1.)
    # check solution as stationary
    drho = L @ rho.reshape(L.shape[0])
    assert np.isclose(np.sum(drho), 0)

L = test_liouvillian(0.1, 5, 0.001)
def test_stationary_solve(row, scale):
    rho = nme.stationary(L, method="solve", row=row, scale=scale)
    # check norm
    tr = np.trace(rho)
    assert np.isclose(tr, 1.)
    # check solution as stationary
    drho = L @ rho.reshape(L.shape[0])
    assert np.isclose(np.sum(drho), 0)

def test_stationary_single_cavity_mode():
    #parameters
    hw_ph = 1
    kappa = 0.1
    kT = 1e-1
    n = np.arange(10)

    a, Nb = sq.composite(boson_modes=1, max_bosons=max(n))
    
    H = hw_ph * Nb
    
    #numerics
    cp, cm = no.collapses(a, H, kT, 'bosonic', kappa, total=False)
    L = no.liouvillian(H, cp + cm)
    Pn = nme.stationary(L).diagonal()

    # photons coming out
    In = nme.current(no.jump(cm) - no.jump(cp), L)
    n_average = nme.current(no.jump(cm), L)

    #analytics
    Gup = kappa * ndist.bose_einstein(hw_ph, kT)
    Gdw = kappa * (1 + ndist.bose_einstein(hw_ph, kT))
    x = Gup / Gdw
    gamma = (1 - x) / (1 - x ** (max(n) + 1))
    Pa = gamma * x ** n
    
    #tr(Jrho) = tr(a rho a^\dagger) = <n>
    n_average_a = kappa * ndist.bose_einstein(hw_ph, kT)
    
    assert np.allclose(Pn, Pa)
    assert np.allclose(In, 0)
    assert np.allclose(n_average, n_average_a)

