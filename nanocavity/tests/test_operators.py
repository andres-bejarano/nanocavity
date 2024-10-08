import numpy as np
import nanocavity.operators as no
import nanocavity.qutip.operators as nqo
import nanocavity.rate_equation as nre
import nanocavity.tls as ntls
import nanocavity.qutip.tls as nqtls
import secondquant as sq

Eg = 0.4
Delta = 0.9
hw_ph = 1
g_ph = 0.3

U = 0
max_bosons = 1

H_parameters = Eg, Delta, hw_ph, g_ph, U, max_bosons

kappa = 0.1
Gamma_L = 1e-3
Gamma_R = 2e-3

VL = 3
VR = -3
kT = 0.1


def test_Htls_nc_QuTiP():
    for rwa in (True, False):
        for max_bosons in (1, 2):
            U = 0
            H_parameters = Eg, Delta, hw_ph, g_ph, U, max_bosons, rwa
            Hnc0, Hnc1, [dg_nc, de_nc, a_nc] = ntls.Hamiltonian(*H_parameters)
            Hqt0, Hqt1, [dg_qt, de_qt, a_qt] = nqtls.Hamiltonian(*H_parameters)
            Hnc = Hnc0 + Hnc1
            Hqt = Hqt0 + Hqt1
            Enc, Vnc = Hnc.eigh()
            Eqt, Vqt = Hqt.eigenstates()

            dg_nc = dg_nc.toarray()
            de_nc = de_nc.toarray()
            a_nc = a_nc.toarray()

            dg_qt = dg_qt.full()
            de_qt = de_qt.full()
            a_qt = a_qt.full()

            assert np.allclose(Hnc.toarray(), Hqt.full())
            assert np.allclose(Enc, Eqt)
            dim = a_nc.shape[0]
            for i in range(dim):
                M = np.vstack([Vqt[i].full().reshape(dim), Vnc[:, i]])
                rank = np.linalg.matrix_rank(M)
                assert rank == 1

            assert np.allclose(
                dg_nc.T @ dg_nc + dg_nc @ dg_nc.T, np.eye(dg_nc.shape[0])
            )
            assert np.allclose(dg_nc.T @ de_nc + de_nc @ dg_nc.T, 0)
            assert np.allclose(de_nc.T @ dg_nc + dg_nc @ de_nc.T, 0)
            assert np.allclose(
                de_nc.T @ de_nc + de_nc @ de_nc.T, np.eye(de_nc.shape[0])
            )
            assert np.allclose(
                a_nc.T @ a_nc - a_nc @ a_nc.T, a_nc.T @ a_nc - a_nc @ a_nc.T
            )

            assert np.allclose(dg_nc, dg_qt)
            assert np.allclose(de_nc, de_qt)
            assert np.allclose(a_nc, a_qt)


def test_collapses():
    g_ph = 0.05
    U = 0
    max_bosons = 1
    Hnc0, Hnc1, [dg_nc, de_nc, a_nc] = ntls.Hamiltonian(
        Eg, Delta, hw_ph, g_ph, U, max_bosons
    )
    Hqt0, Hqt1, [dg_qt, de_qt, a_qt] = nqtls.Hamiltonian(
        Eg, Delta, hw_ph, g_ph, U, max_bosons
    )
    Hnc = Hnc0 + Hnc1
    Hqt = Hqt0 + Hqt1
    for VL, VR in [[-3, 3], [-1, 3], [0, 3], [1, 3]]:
        Cqt = nqtls.collapses(
            Hqt, [dg_qt, de_qt, a_qt], VL, VR, kappa, Gamma_L, Gamma_R, kT
        )
        Cnc = ntls.collapses(
            Hnc, [dg_nc, de_nc, a_nc], VL, VR, kappa, Gamma_L, Gamma_R, kT
        )

        for i in range(len(Cnc)):
            assert np.allclose(Cnc[i], Cqt[i].full())


def test_jump_operator():
    U = 0
    max_bosons = 1
    H_parameters = Eg, Delta, hw_ph, g_ph, U, max_bosons
    Hnc0, Hnc1, [dg_nc, de_nc, a_nc] = ntls.Hamiltonian(*H_parameters)
    Hqt0, Hqt1, [dg_qt, de_qt, a_qt] = nqtls.Hamiltonian(*H_parameters)
    Hnc = Hnc0 + Hnc1
    Hqt = Hqt0 + Hqt1
    for VL, VR in [[-3, 3], [-1, 3], [0, 3], [1, 3]]:
        c_qt = nqtls.collapses(
            Hqt, [dg_qt, de_qt, a_qt], VL, VR, kappa, Gamma_L, Gamma_R, kT
        )
        c_nc = ntls.collapses(
            Hnc, [dg_nc, de_nc, a_nc], VL, VR, kappa, Gamma_L, Gamma_R, kT
        )

        Jqt = nqo.jump(c_qt)
        Jnc = no.jump(c_nc)

        assert np.allclose(Jqt.full(), Jnc)


def test_dissipators():
    g_ph = 0.05
    for VL, VR in [[-3, 3], [-1, 3], [0, 3], [1, 3]]:
        U = 0
        max_bosons = 1
        Hnc0, Hnc1, [dg_nc, de_nc, a_nc] = ntls.Hamiltonian(
            Eg, Delta, hw_ph, g_ph, U, max_bosons
        )
        Hqt0, Hqt1, [dg_qt, de_qt, a_qt] = nqtls.Hamiltonian(
            Eg, Delta, hw_ph, g_ph, U, max_bosons
        )
        Hnc = Hnc0 + Hnc1
        Hqt = Hqt0 + Hqt1
        c_ops_qt = list(
            nqtls.collapses(
                Hqt, [dg_qt, de_qt, a_qt], VL, VR, kappa, Gamma_L, Gamma_R, kT
            )
        )
        c_ops_nc = ntls.collapses(
            Hnc, [dg_nc, de_nc, a_nc], VL, VR, kappa, Gamma_L, Gamma_R, kT
        )

        Dnc = no.dissipator(c_ops_nc)
        Dqo = nqo.dissipator(c_ops_qt)
        Dqt = nqo.dissipator(c_ops_qt, lindblad=True)
        assert np.allclose(Dqo.full(), Dqt.full())
        assert np.allclose(Dnc, Dqt.full())


def test_liovillian():
    for coupling in [0.005, 0.05, 0.5]:
        U = 0
        max_bosons = 1
        H_parameters = Eg, Delta, hw_ph, coupling, U, max_bosons
        Hqt0, Hqt1, [dg_qt, de_qt, a_qt] = nqtls.Hamiltonian(*H_parameters)
        Hnc0, Hnc1, [dg_nc, de_nc, a_nc] = ntls.Hamiltonian(*H_parameters)
        Hqt = Hqt0 + Hqt1
        Hnc = Hnc0 + Hnc1
        for VL, VR in [[-3, 3], [-1, 3], [0, 3], [1, 3]]:
            c_ops_qt = list(
                nqtls.collapses(
                    Hqt, [dg_qt, de_qt, a_qt], VL, VR, kappa, Gamma_L, Gamma_R, kT
                )
            )
            Lqt = nqo.liouvillian(Hqt, c_ops_qt)

            c_ops_nc = ntls.collapses(
                Hnc, [dg_nc, de_nc, a_nc], VL, VR, kappa, Gamma_L, Gamma_R, kT
            )
            Lnc = no.liouvillian(Hnc, c_ops_nc)
            assert np.allclose(Lnc, Lqt.full())


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
