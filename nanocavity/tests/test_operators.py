import numpy as np
import nanocavity.operators as no
import nanocavity.tls as ntls
import secondquant as sq
import pytest
import nanocavity.distributions as nd


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
    c_ops = no.collapses(c, H, kT, "fermionic", rate)
    a_ops = no.collapses(a, H, kT, "bosonic", rate)
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
    OPS = no.collapses(a, H0, kT, "bosoNIC", kappa)

    # collapses using X+ can be called in different ways
    ops1 = no.collapses(a, H0, kT, "bosonicX+", kappa)
    ops2 = no.collapses(a, H0, kT, "X+bosonic", kappa)
    ops3 = no.collapses(a, H0, kT, "bosonic-X+", kappa)
    ops4 = no.collapses(a, H0, kT, "bosonicx+", kappa)
    # with hw=1 the collapses are identical with a or X+
    for ops in (ops1, ops2, ops3, ops4):
        assert np.allclose(OPS, ops)

    # the story is different with dressed rates
    ops1 = no.collapses(a, H0 + Hint, kT, "bosoNIC", kappa)
    ops2 = no.collapses(a, H0 + Hint, kT, "bosonic-x+", kappa)
    assert len(ops1) != len(ops2)

    # changing cavity mode energy also introduces differences
    H0, Hint, [dg, de, a] = ntls.Hamiltonian(Eg, Delta, 0.5, g_ph, 0, 1)
    ops1 = no.collapses(a, H0, kT, "Bosonic", kappa)
    ops2 = no.collapses(a, H0, kT, "bosonicX+", kappa)
    assert not np.allclose(ops1, ops2)

    # the dependency on hw_ph can be incorporated as a renormalized rate
    ops3 = no.collapses(a, H0, kT, "bosonicX+", kappa / 0.5**0.5)
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
    cp, cm = no.collapses(dg, H, kT, "fermionic", Gamma, V, False)
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
    cp, cm = no.collapses(ag, H, kT, "bosonic", kappa, total=False)
    cp_an = np.zeros((2, 2))
    cm_an = np.zeros((2, 2))
    cp_an[1, 0] = np.sqrt(kappa * nd.bose_einstein(hw, kT))
    cm_an[0, 1] = np.sqrt(kappa * (1 - nd.bose_einstein(hw, kT)))
    assert np.allclose(cp, cp_an)
    assert np.allclose(cm, cm_an)


def test_coherent_evolution():
    Delta = 1
    H = Delta / 2 * np.array([[1, 0], [0, -1]])
    L = no.liouvillian(H)
    L_analytics = np.zeros((4, 4), dtype="complex")
    L_analytics[1, 1] = -1j * Delta
    L_analytics[2, 2] = 1j * Delta
    assert np.allclose(L, L_analytics)
