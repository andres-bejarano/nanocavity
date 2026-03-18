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
    bin_width = 1e-6
    cp, cm = no.collapses(c, basis, kT, "fermionic", rate, bin_width)
    ap, am = no.collapses(a, basis, kT, "bosonic", rate, bin_width)
    collapse_ops = [cp, cm, ap, am]
    for c_ops in collapse_ops:
        for ops in c_ops.values():
            D1 = no.dissipator(ops)
            D2 = no.dissipator(ops, method="einsum")
            assert np.allclose(D1, D2)
    L1 = no.liouvillian(H, collapse_ops)
    L2 = no.liouvillian(H, collapse_ops, method="einsum")
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
    bin_width = 1e-6
    OPSp, OPSm = no.collapses(a, basis, kT, "bosoNIC", kappa, bin_width)

    # collapses using X+ can be called in different ways
    ops1p, ops1m = no.collapses(a, basis, kT, "bosonicX+", kappa, bin_width)
    ops2p, ops2m = no.collapses(a, basis, kT, "X+bosonic", kappa, bin_width)
    ops3p, ops3m = no.collapses(a, basis, kT, "bosonic-X+", kappa, bin_width)
    ops4p, ops4m = no.collapses(a, basis, kT, "bosonicx+", kappa, bin_width)
    # with hw=1 the collapses are identical with a or X+
    for Ediff in OPSp:
        for ops in [ops1p, ops2p, ops3p, ops4p]:
            assert np.allclose(OPSp[Ediff], ops[Ediff])
    # for Ediff in OPSm:
    #     for ops in [ops1m, ops2m, ops3m, ops4m]:
    #         assert np.allclose(OPSp[Ediff], ops[Ediff])

    # the story is different with dressed rates
    basis = (H0 + Hint).eigh()
    ops1p, ops1m = no.collapses(a, basis, kT, "bosoNIC", kappa, bin_width)
    ops2p, ops2m = no.collapses(a, basis, kT, "bosonic-x+", kappa, bin_width)
    assert ops1p.keys() != ops2p.keys()
    assert ops1m.keys() != ops2m.keys()
    #
    # # changing cavity mode energy also introduces differences
    H0, Hint, [dg, de, a] = ntls.Hamiltonian(Eg, Delta, 0.5, g_ph, 0, 1)
    basis = H0.eigh()
    ops1p, ops1m = no.collapses(a, basis, kT, "Bosonic", kappa, bin_width)
    ops2p, ops2m = no.collapses(a, basis, kT, "bosonicX+", kappa, bin_width)
    for key in ops1p.keys():
        assert not np.allclose(ops1p[key], ops2p[key])
    for key in ops1m.keys():
        assert not np.allclose(ops1m[key], ops2m[key])
    #
    # # the dependency on hw_ph can be incorporated as a renormalized rate
    ops3p, ops3m = no.collapses(a, basis, kT, "bosonicX+", kappa / 0.5**0.5, bin_width)
    assert not np.allclose(list(OPSp.keys()), list(ops3p.keys()))
    assert not np.allclose(list(OPSm.keys()), list(ops3m.keys()))


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
    bin_width = 1e-6
    cp, cm = no.collapses(dg, basis, kT, "fermionic", Gamma, bin_width, V)
    cp_an = np.zeros((2, 2))
    cm_an = np.zeros((2, 2))
    cp_an[1, 0] = np.sqrt(Gamma * nd.fermi_dirac(Eg, kT, V))
    cm_an[0, 1] = np.sqrt(Gamma * (1 - nd.fermi_dirac(Eg, kT, V)))
    assert np.allclose(list(cp.values()), cp_an)
    assert np.allclose(list(cm.values()), cm_an)


def test_bosonic_collapse(kappa):
    hw = 1
    kT = 1e-2
    ag, ng = sq.composite(fermion_modes=0, boson_modes=[1])
    H = hw * ng
    basis = H.eigh()
    bin_width = 1e-6
    cp, cm = no.collapses(ag, basis, kT, "bosonic", kappa, bin_width)
    cp_an = np.zeros((2, 2))
    cm_an = np.zeros((2, 2))
    cp_an[1, 0] = np.sqrt(kappa * nd.bose_einstein(hw, kT))
    cm_an[0, 1] = np.sqrt(kappa * (1 - nd.bose_einstein(hw, kT)))
    assert np.allclose(list(cp.values()), cp_an)
    assert np.allclose(list(cm.values()), cm_an)


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
    bin_width = 1e-6
    c_opsp, c_opsm = no.collapses(a, basis, kT, "bosonic", kappa, bin_width)
    c_ops = [c_opsp, c_opsm]

    for method in [None, "einsum", "kron"]:
        dissipator = 0
        if method is None:
            for c in c_ops:
                for k in c.keys():
                    dissipator += no.dissipator(c[k])
        else:
            for c in c_ops:
                for k in c.keys():
                    dissipator += no.dissipator(c[k], method=method)
        assert np.allclose(dissipator, D_ana)


def test_jump():
    Delta = 1
    kT = 1e-2
    kappa = 1e-3
    a, n = sq.composite(boson_modes=[1])
    H = Delta * n
    basis = H.eigh()
    bin_width = 1e-6
    c_opsp, c_opsm = no.collapses(a, basis, kT, "bosonic", kappa, bin_width)
    c_ops = [c_opsp, c_opsm]
    dim = a.shape[0]
    J = 0
    for c_dict in c_ops:
        for k in c_dict.keys():
            J += no.jump(c_dict[k])
    J_ana = np.zeros((4, 4))
    J_ana[0, 3] = kappa
    assert np.allclose(J, J_ana)

    for op in (a, a.toarray()):
        J1 = kappa * no.jump(op)
        assert np.allclose(J1, J_ana)


def test_damped_cavity():
    hw_ph = 1
    a, n = sq.composite(boson_modes=[1])
    H = hw_ph * n
    basis = H.eigh()
    bin_width = 1e-6
    kT = 0.01
    kappa = 0.1
    c_opsp, c_opsm = no.collapses(a, basis, kT, "bosonic", kappa, bin_width)
    c_ops = [c_opsp, c_opsm]

    dissipator = 0
    for c in c_ops:
        for k in c.keys():
            dissipator += no.dissipator(c[k])

    c_ops_direct = [
        {
            "hw_ph": [np.sqrt(kappa * nd.bose_einstein(hw_ph, kT)) * a.d.toarray()],
            "-hw_ph": [
                np.sqrt(kappa * (1 - nd.bose_einstein(hw_ph, kT))) * a.toarray()
            ],
        }
    ]

    dissipator_d = 0
    for c in c_ops_direct:
        for k in c.keys():
            dissipator_d += no.dissipator(c[k])
    assert np.allclose(dissipator, dissipator_d)
