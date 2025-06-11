import nanocavity.ols as nols
import nanocavity.operators as no
import nanocavity.master_equation as nme
import nanocavity.franck_condon as nfc
import nanocavity.distributions as nd
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

    Gamma_s = 1e-4
    Gamma_t = 5e-4
    kappa = 0.1
    kT = 1e-3
    g_ph = 0.5
    Vs = -0.5
    Vt = 1.5

    L = nols.liouvillian(dg, a_ph, Hs, g_ph, Vs, Vt, Gamma_s, Gamma_t, kappa, kT)

    assert np.allclose(L.shape[0], Hs.toarray().shape[0] ** 2)
    assert np.allclose(L.shape[1], Hs.toarray().shape[1] ** 2)
    assert np.allclose(L[0, 7], kappa)



def test_populations():
    hw_ph = 1
    max_bosons = 5
    Hs, [dg, a_ph], [ng, n_ph] = nols.Hamiltonian(hw_ph, max_bosons=max_bosons)

    kappa = 0.1
    kT = 0.01



    Vt = 0.5
    Vs = -1.5
    Gamma_t = 1e-6
    Gamma_s = 5e-6

    g_ph = 0.5

    for VL, VR in [[1.5, 0.5], [2.5, -0.5], [1.5, -1.5]]:
        L = nols.liouvillian(dg, a_ph, Hs, g_ph, Vs, Vt, Gamma_s, Gamma_t, kappa, kT)
        rho_st = np.real(nme.stationary(L))
        index = nols.get_diagonal_indices(max_bosons=max_bosons, cutoff=2)

        row, colum = zip(*index)
        Pnum = rho_st[list(row), list(colum)]
        Pan = nols.Pst(Vt, Vs, Gamma_t, Gamma_s, g_ph, kappa)
        assert np.allclose(Pnum, Pan)