import nanocavity.ols as ols
import nanocavity.operators as no
from itertools import chain
from secondquant.operator import Operator
import numpy as np
import secondquant as sq


def test_hamiltonian():
    Hs, anni_ops, num_ops = ols.Hamiltonian(1)
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
    Hs, [D, a_ph], [ng, n_ph] = ols.Hamiltonian(hw_ph)
    D, A = ols.Lang_Firsov_transform(D, a_ph, g_ph)
    assert isinstance(D, Operator)
    assert isinstance(A, Operator)


def test_collapse_electronic():
    hw_ph = 1
    g_ph = 1
    Hs, [D, a_ph], [ng, n_ph] = ols.Hamiltonian(hw_ph)
    D, A = ols.Lang_Firsov_transform(D, a_ph, g_ph)
    basis = Hs.eigh()

    coll_list = ols.collapse_electronic(D, basis, 1, -1, 1e-4, 5e-4, 1e-3)
    assert isinstance(coll_list[0], list)
    assert isinstance(coll_list, tuple)
    assert len(coll_list) == 4
    coll_tot = ols.collapse_electronic(D, basis, 1, -1, 1e-4, 5e-4, 1e-3, total=True)
    assert len(coll_tot) == len(list(chain.from_iterable(coll_list)))


def test_collapse_dephasing():
    hw_ph = 1
    g_ph = 1
    kappa = 0.1
    Hs, [D, a_ph], [ng, n_ph] = ols.Hamiltonian(hw_ph, max_bosons=5)
    D, A = ols.Lang_Firsov_transform(D, a_ph, g_ph)
    basis = Hs.eigh()
    coll_list = ols.collapse_dephasing(ng, basis, kappa, g_ph)
    assert len(coll_list) == 144
    assert isinstance(coll_list[0], np.ndarray)


def test_dissipator():
    Delta = 1
    kT = 1e-2
    kappa = 1e-3
    D_ana = np.zeros((4, 4))
    D_ana[1, 1] = -1
    D_ana[2, 2] = -1
    D_ana[3, 3] = -2
    D_ana[0, 3] = 2
    D_ana *= kappa / 2

    a, n = sq.composite(boson_modes=[1])
    H = Delta * n
    basis = H.eigh()
    c_ops = no.collapses(a, basis, kT, "bosonic", kappa)

    for method in [None, "einsum", "kron"]:
        if method is None:
            dissipator = ols.dissipator(c_ops)
        else:
            dissipator = ols.dissipator(c_ops, method=method)
        assert np.allclose(dissipator, D_ana)


def test_liouvillian():
    hw_ph = 1
    Hs, [dg, a_ph], [ng, n_ph] = ols.Hamiltonian(hw_ph, max_bosons=5)

    Gamma_s = 5e-4
    Gamma_t = 1e-4
    kappa = 0.1
    kT = 1e-3
    g_ph = 0.5
    Vs = 1.5
    Vt = -0.5

    L = ols.liouvillian(dg, a_ph, Hs, g_ph, Vs, Vt, Gamma_s, Gamma_t, kappa, kT)

    assert np.allclose(L.shape[0], Hs.toarray().shape[0] ** 2)
    assert np.allclose(L.shape[1], Hs.toarray().shape[1] ** 2)
