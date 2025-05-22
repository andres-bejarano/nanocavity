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

def test_get_diagonal_indices():
    Nmax = 5 
    index = nols.get_diagonal_indices(max_bosons=Nmax, cutoff=2)

    assert len(index) == 6
    for i in range(3):
        assert index[i] == [i, i]
        assert index[i + 3] == [Nmax + 1 + i, Nmax + 1 + i]

def get_basis_index(n, q, max_bosons):

    max_bosons = 4
    for n in range(max_bosons + 1):
        result = get_basis_index(n, 0, max_bosons)
        expected = n
        assert result == expected

    # Test for q = 1
    for n in range(max_bosons + 1):
        result = get_basis_index(n, 1, max_bosons)
        expected = max_bosons + 1 + n
        assert result == expected

    # Test for invalid charge
    try:
        get_basis_index(0, 2, max_bosons)
        assert False, "Expected ValueError for invalid charge q=2"
    except ValueError:
        pass  # This is the expected behavior

def test_get_vectorized_rho_index():
    # we check population index
    N_ph_max = 4
    index_h, index_e = [], []
    for i in range(3):
        index_h.append(nols.get_vectorized_rho_index(n1=i, q1=0, n2=i, q2=0, max_bosons=N_ph_max))
        index_e.append(nols.get_vectorized_rho_index(n1=i, q1=1, n2=i, q2=1, max_bosons=N_ph_max))
    index = index_h +  index_e

    # index_by_hand
    dim = 2 * (N_ph_max + 1)

    i_h0 = 0
    i_h1 = dim + 1
    i_h2 = i_h1 + dim + 1

    i_e0 = (N_ph_max + 1) *  dim + (N_ph_max + 1)
    i_e1 = i_e0 + dim + 1
    i_e2 = i_e1 + dim + 1

    index2 = [i_h0, i_h1, i_h2, i_e0, i_e1, i_e2]

    assert index == index2 
