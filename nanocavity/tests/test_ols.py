import nanocavity.ols as nols
import nanocavity.operators as no
from itertools import chain
from secondquant.operator import Operator
import numpy as np
import secondquant as sq


def test_hamiltonian():
    Hs, anni_ops, num_ops = nols.Hamiltonian(1)
    assert isinstance(Hs, Operator)
    assert isinstance(anni_ops, list)
    assert isinstance(anni_ops[0], Operator)
    assert isinstance(anni_ops[1], Operator)
    assert isinstance(num_ops, list)
    assert isinstance(num_ops[0], Operator)
    assert isinstance(num_ops[1], Operator)


def test_lang_firsov_transform():
    hw_ph = 1
    g_ph = 1
    Hs, [D, a_ph], [ng, n_ph] = nols.Hamiltonian(hw_ph)
    D, A = nols.Lang_Firsov_transform(D, a_ph, g_ph)
    assert isinstance(D, Operator)
    assert isinstance(A, Operator)


def test_collapse_electronic():
    hw_ph = 1
    g_ph = 1
    Hs, [D, a_ph], [ng, n_ph] = nols.Hamiltonian(hw_ph)
    D, A = nols.Lang_Firsov_transform(D, a_ph, g_ph)
    basis = Hs.eigh()

    coll_list = nols.collapse_electronic(D, basis, 1, -1, 1e-4, 5e-4, 1e-3)
    assert isinstance(coll_list[0], list)
    assert isinstance(coll_list, tuple)
    assert len(coll_list) == 2  # Plus, Minus


def test_liouvillian():
    hw_ph = 1
    Hs, [dg, a_ph], [ng, n_ph] = nols.Hamiltonian(hw_ph, max_bosons=2)

    Gamma_s = 5e-4
    Gamma_t = 1e-4
    kappa = 0.1
    kT = 1e-3
    g_ph = 0.5
    Vs = 1.5
    Vt = -0.5

    L = nols.liouvillian(dg, a_ph, Hs, g_ph, Vs, Vt, Gamma_s, Gamma_t, kappa, kT)

    assert np.allclose(L.shape[0], Hs.toarray().shape[0] ** 2)
    assert np.allclose(L.shape[1], Hs.toarray().shape[1] ** 2)
    assert np.allclose(L[0, 7], kappa)

def test_G_p_nm_a():
    assert np.isclose(nols.G_p_nm_a(1, 1, 1, 0.5, 1e-4, 0.001), 0)

    assert np.isclose(nols.G_p_nm_a(3, 1, 0.3, 0.5, 1e-4, 0.001), 0)
    assert not np.isclose(nols.G_p_nm_a(3, 1, 0.3, 2.5, 1e-4, 0.001), 0)

    assert not np.isclose(nols.G_p_nm_a(1, 3, 0.3, 0.5, 1e-4, 0.001), 0)
    assert not np.isclose(nols.G_p_nm_a(1, 3, 0.3, 2.5, 1e-4, 0.001), 0)

def test_G_m_nm_a():
    assert np.isclose(nols.G_m_nm_a(1, 1, 1, 0.5, 1e-4, 0.001), 0)

    assert np.isclose(nols.G_m_nm_a(3, 1, 0.3, 0.5, 1e-4, 0.001), 0)
    assert np.isclose(nols.G_m_nm_a(3, 1, 0.3, 2.5, 1e-4, 0.001), 0)

    assert np.isclose(nols.G_m_nm_a(1, 3, 0.3, 2.5, 1e-4, 0.001), 0)
    assert not np.isclose(nols.G_m_nm_a(1, 3, 0.3, 0.5, 1e-4, 0.001), 0)
