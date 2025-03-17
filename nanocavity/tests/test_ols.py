import nanocavity.ols as ols
from itertools import chain
from secondquant.operator import Operator


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

    c_pL, c_mL, c_pR, c_mR = ols.collapse_electronic(D, basis, 1, -1, 1e-4, 5e-4, 1e-3)
    coll_list = [c_pL, c_mL, c_pR, c_mR]
    for c in coll_list:
        assert isinstance(c, list)
    assert len(coll_list) == 4
    coll_tot = ols.collapse_electronic(D, basis, 1, -1, 1e-4, 5e-4, 1e-3, total=True)
    assert len(coll_tot) == len(list(chain.from_iterable(coll_list)))
