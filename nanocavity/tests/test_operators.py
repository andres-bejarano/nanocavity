import numpy as np
import pytest
import secondquant as sq

import nanocavity.distributions as nd
import nanocavity.operators as no
import nanocavity.tls as ntls
import nanocavity.ols as nols


def test_liouvillian_basic():
    c, n = sq.composite(fermion_modes=3)
    # building some Hamiltonian
    H = 0
    for i in range(len(c) - 1):
        cdc = c[i].d * c[i + 1]
        H += cdc + cdc.d
    for i, ni in enumerate(n):
        H += (i + 1) * ni
    # let's build a random density matrix
    rho = np.random.rand(*H.shape)
    rho_op = sq.operator.Operator(rho)
    # compute -i[H, rho]
    drho = -1j * sq.commutator(H, rho_op)
    for method in [None, "einsum", "kron"]:
        # construct Liouvillian
        if method is None:
            L = no.liouvillian(H)
        else:
            L = no.liouvillian(H, method=method)
        # apply it to rho after reshape to vector
        drho2 = L @ rho.reshape(len(L))
        # and reshape back to matrix
        drho2 = drho2.reshape(H.shape)
        assert np.allclose(drho.toarray(), drho2)
        assert not np.allclose(drho.toarray(), -drho2)


def test_einsum():
    [c, a], [Nf, Nb] = sq.composite(fermion_modes=1, boson_modes=1, max_bosons=3)
    H = Nf + 0.1 * Nb + 0.01 * (c.d * a + a.d * c)
    rate = 0.1
    kT = 0.01
    basis = H.eigh()
    cp, cm = no.collapses(c, basis, kT, "fermionic", rate)
    c_ops = cp + cm
    ap, am = no.collapses(a, basis, kT, "bosonic", rate)
    a_ops = ap + am
    for ops in (c_ops, a_ops):
        D1 = no.dissipator(ops)
        D2 = no.dissipator(ops, method="einsum")
        assert np.allclose(D1, D2)
    L1 = no.liouvillian(H, c_ops + a_ops)
    L2 = no.liouvillian(H, c_ops + a_ops, method="einsum")
    assert np.allclose(L1, L2)


def test_bosonic_collapses():
    # hw_ph = 1
    Eg = 0.4
    Delta = 0.9
    g_ph = 0.3
    kappa = 0.1

    kT = 0.1
    H0, Hint, [dg, de, a] = ntls.Hamiltonian(Eg, Delta, 1.0, g_ph, 0, 1)

    basis = H0.eigh()
    OPSp, OPSm = no.collapses(a, basis, kT, "bosoNIC", kappa)
    OPS = OPSp + OPSm

    # collapses using X+ can be called in different ways
    ops1p, ops1m = no.collapses(a, basis, kT, "bosonicX+", kappa)
    ops1 = ops1p + ops1m
    ops2p, ops2m = no.collapses(a, basis, kT, "X+bosonic", kappa)
    ops2 = ops2p + ops2m
    ops3p, ops3m = no.collapses(a, basis, kT, "bosonic-X+", kappa)
    ops3 = ops3p + ops3m
    ops4p, ops4m = no.collapses(a, basis, kT, "bosonicx+", kappa)
    ops4 = ops4p + ops4m
    # with hw=1 the collapses are identical with a or X+
    for ops in (ops1, ops2, ops3, ops4):
        assert np.allclose(OPS, ops)

    # the story is different with dressed rates
    basis = (H0 + Hint).eigh()
    ops1p, ops1m = no.collapses(a, basis, kT, "bosoNIC", kappa)
    ops1 = ops1p + ops1m
    ops2p, ops2m = no.collapses(a, basis, kT, "bosonic-x+", kappa)
    ops2 = ops2p + ops2m
    assert len(ops1) != len(ops2)

    # changing cavity mode energy also introduces differences
    H0, Hint, [dg, de, a] = ntls.Hamiltonian(Eg, Delta, 0.5, g_ph, 0, 1)
    basis = H0.eigh()
    ops1p, ops1m = no.collapses(a, basis, kT, "Bosonic", kappa)
    ops1 = ops1p + ops1m
    ops2p, ops2m = no.collapses(a, basis, kT, "bosonicX+", kappa)
    ops2 = ops2p + ops2m
    assert not np.allclose(ops1, ops2)

    # the dependency on hw_ph can be incorporated as a renormalized rate
    ops3p, ops3m = no.collapses(a, basis, kT, "bosonicX+", kappa / 0.5**0.5)
    ops3 = ops3p + ops3m
    assert not np.allclose(OPS, ops3)


@pytest.fixture(scope="module", params=[-0.5, 0.5])
def Eg(request):
    return request.param


@pytest.fixture(scope="module", params=[-0.5, 0, 0.5])
def V(request):
    return request.param


@pytest.fixture(scope="module", params=[1e-3, 1e-2])
def kappa(request):
    return request.param


def test_fermionic_collapse(Eg, V):
    Gamma = 1e-3
    dg, ng = sq.composite(1)
    H = Eg * dg.d * dg
    kT = 1e-2
    basis = H.eigh()
    cp, cm = no.collapses(dg, basis, kT, "fermionic", Gamma, V, False)
    cp_an = np.zeros((2, 2))
    cm_an = np.zeros((2, 2))
    cp_an[1, 0] = np.sqrt(Gamma * nd.fermi_dirac(Eg, kT, V))
    cm_an[0, 1] = np.sqrt(Gamma * (1 - nd.fermi_dirac(Eg, kT, V)))
    assert np.allclose(cp, cp_an)
    assert np.allclose(cm, cm_an)


def test_bosonic_collapse(kappa):
    hw = 1
    kT = 1e-2
    ag, ng = sq.composite(fermion_modes=0, boson_modes=[1])
    H = hw * ng
    basis = H.eigh()
    cp, cm = no.collapses(ag, basis, kT, "bosonic", kappa)
    cp_an = np.zeros((2, 2))
    cm_an = np.zeros((2, 2))
    cp_an[1, 0] = np.sqrt(kappa * nd.bose_einstein(hw, kT))
    cm_an[0, 1] = np.sqrt(kappa * (1 - nd.bose_einstein(hw, kT)))
    assert np.allclose(cp, cp_an)
    assert np.allclose(cm, cm_an)


def test_coherent_evolution():
    Delta = 1
    H = Delta / 2 * np.array([[1, 0], [0, -1]])
    L_analytics = np.zeros((4, 4), dtype="complex")
    L_analytics[1, 1] = -1j * Delta
    L_analytics[2, 2] = 1j * Delta
    for method in [None, "einsum", "kron"]:
        if method is None:
            L = no.liouvillian(H)
        else:
            L = no.liouvillian(H, method=method)
        assert np.allclose(L, L_analytics)


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
    c_opsp, c_opsm = no.collapses(a, basis, kT, "bosonic", kappa)
    c_ops = c_opsp + c_opsm

    for method in [None, "einsum", "kron"]:
        if method is None:
            dissipator = no.dissipator(c_ops)
        else:
            dissipator = no.dissipator(c_ops, method=method)
        assert np.allclose(dissipator, D_ana)


def test_jump():
    Delta = 1
    kT = 1e-2
    kappa = 1e-3
    a, n = sq.composite(boson_modes=[1])
    H = Delta * n
    basis = H.eigh()
    c_opsp, c_opsm = no.collapses(a, basis, kT, "bosonic", kappa)
    c_ops = c_opsp + c_opsm
    J = no.jump(c_ops)
    J_ana = np.zeros((4, 4))
    J_ana[0, 3] = kappa
    assert np.allclose(J, J_ana)

    for op in (a, a.toarray()):
        J1 = kappa * no.jump(op)
        assert np.allclose(J1, J_ana)


def test_eigenstate_decomp():
    hw_ph = 1
    g_ph = 0.1
    Hs, [dg, a_ph], [ng, n_ph] = nols.Hamiltonian(hw_ph, max_bosons=1)
    basis = Hs.eigh()
    Dg, A = nols.Lang_Firsov_transform(dg, a_ph, g_ph)

    Dg_comp = no.eigenstate_decomp(Dg, basis)
    A_comp = no.eigenstate_decomp(A, basis)
    DgA_comp = no.eigenstate_decomp(Dg + A, basis)

    assert Dg.allclose(sum(Dg_comp))
    assert A.allclose(sum(A_comp))
    assert (Dg + A).allclose(sum(DgA_comp))


def test_full_diss():
    hw_ph = 1
    g_ph = 0.1
    Hs, [dg, a_ph], [ng, n_ph] = nols.Hamiltonian(hw_ph, max_bosons=1)
    basis = Hs.eigh()
    Dg, A = nols.Lang_Firsov_transform(dg, a_ph, g_ph)
    cp = no.eigenstate_decomp(Dg.d, basis)
    cm = no.eigenstate_decomp(Dg, basis)
    c_decomp_dag = sum(cp)
    c_decomp = sum(cm)

    assert Dg.allclose(c_decomp)
    assert Dg.d.allclose(c_decomp_dag)

    diss_eigen = no.dissipator(cp, diagonal_form=False)
    diss_eigen += no.dissipator(cm, diagonal_form=False)

    c_full = [Dg.toarray(), Dg.d.toarray()]
    diss_full = no.dissipator(c_full, diagonal_form=True)

    assert np.allclose(diss_eigen, diss_full)
