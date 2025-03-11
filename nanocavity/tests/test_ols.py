import nanocavity.ols as ols
import secondquant as sq


def test_hamiltonian():
    Hs, anni_ops, num_ops = ols.Hamiltonian(1)
    assert isinstance(Hs, sq.operator.operator.Operator)
    assert isinstance(anni_ops, list)
    assert isinstance(anni_ops[0], sq.operator.operator.Operator)
    assert isinstance(anni_ops[1], sq.operator.operator.Operator)
    assert isinstance(num_ops, list)
    assert isinstance(num_ops[0], sq.operator.operator.Operator)
    assert isinstance(num_ops[1], sq.operator.operator.Operator)


def test_lang_firsov_transform():
    hw_ph = 1
    g_ph = 1
    Hs, [dg, a_ph], [ng, n_ph] = ols.Hamiltonian(hw_ph)
    Dg = ols.Lang_Firsov_transform(dg, a_ph, g_ph)
    assert isinstance(Dg, sq.operator.operator.Operator)


def test_collapse_electronic():
    hw_ph = 1
    g_ph = 1
    Hs, [dg, a_ph], [ng, n_ph] = ols.Hamiltonian(hw_ph)
    Dg = ols.Lang_Firsov_transform(dg, a_ph, g_ph)
    basis = Hs.eigh()

    coll_list = ols.collapse_electronic(Dg, basis, 1, -1, 1e-4, 5e-4, 1e-3)
    assert isinstance(coll_list, list)
    assert isinstance(coll_list[0], list)
    assert len(coll_list) == 4
    coll_tot = ols.collapse_electronic(Dg, basis, 1, -1, 1e-4, 5e-4, 1e-3, total=True)
    assert len(coll_tot) != 4
