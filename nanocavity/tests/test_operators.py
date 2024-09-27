import numpy as np
import nanocavity.operators as no
import nanocavity.qutip.operators as nqo
import nanocavity.rate_equation as nre
import nanocavity.tls as ntls
import nanocavity.qutip.tls as nqtls


Eg = 0.4
delta = 0.9
omegac = 1
coupling = 0.3

H_parameters = Eg, delta, omegac, coupling

kappa = 0.1
gL = 1e-3
gR = 2e-3

VL = 3
VR = -3
kT = 0.1


def test_Htls_nc_QuTiP():
    for rwa in (True, False):
        for max_bosons in (1, 2):
            H_parameters = Eg, delta, omegac, coupling, rwa, max_bosons
            Hnc0, Hnc1, [Dg, De, A] = ntls.Hamiltonian(*H_parameters)
            Hqt0, Hqt1, [dg, de, a] = nqtls.Hamiltonian(*H_parameters)
            Hnc = Hnc0 + Hnc1
            Hqt = Hqt0 + Hqt1
            Enc, Vnc = Hnc.eigh()
            Eqt, Vqt = Hqt.eigenstates()

            Dg = Dg.toarray()
            De = De.toarray()
            A = A.toarray()

            dg = dg.full()
            de = de.full()
            a = a.full()

            assert np.allclose(Hnc.toarray(), Hqt.full())
            assert np.allclose(Enc, Eqt)
            dim = A.shape[0]
            for i in range(dim):
                M = np.vstack([Vqt[i].full().reshape(dim), Vnc[:, i]])
                rank = np.linalg.matrix_rank(M)
                assert rank == 1

            assert np.allclose(Dg.T @ Dg + Dg @ Dg.T, np.eye(Dg.shape[0]))
            assert np.allclose(Dg.T @ De + De @ Dg.T, 0)
            assert np.allclose(De.T @ Dg + Dg @ De.T, 0)
            assert np.allclose(De.T @ De + De @ De.T, np.eye(De.shape[0]))
            assert np.allclose(A.T @ A - A @ A.T, a.T @ a - a @ a.T)

            assert np.allclose(Dg, dg)
            assert np.allclose(De, de)
            assert np.allclose(A, a)


def test_collapses():
    for coupling in [0.005, 0.05, 0.5]:
        H_parameters = Eg, delta, omegac, coupling
        for VL, VR in [[-3, 3], [-1, 3], [0, 3], [1, 3]]:
            Cqt = nqtls.collapses(H_parameters, VL, VR, kappa, gL, gR, kT)
            Cnc = ntls.collapses(H_parameters, VL, VR, kappa, gL, gR, kT)

            for i in range(len(Cnc)):
                assert np.allclose(Cnc[i], Cqt[i].full())


def test_jump_operator():
    for VL, VR in [[-3, 3], [-1, 3], [0, 3], [1, 3]]:
        c_qt = nqtls.collapses(H_parameters, VL, VR, kappa, gL, gR, kT)
        c_nc = ntls.collapses(H_parameters, VL, VR, kappa, gL, gR, kT)

        Jqt = nqo.jump(c_qt)
        Jnc = no.jump(c_nc)

        assert np.allclose(Jqt.full(), Jnc)


def test_dissipators():
    for coupling in [0.005, 0.05, 0.5]:
        for VL, VR in [[-3, 3], [-1, 3], [0, 3], [1, 3]]:
            H_parameters = Eg, delta, omegac, coupling
            c_ops_qt = list(
                    nqtls.collapses(H_parameters, VL, VR, kappa, gL, gR, kT)
            )
            c_ops_nc = ntls.collapses(H_parameters, VL, VR, kappa, gL, gR, kT)

            Dnc = no.dissipator(c_ops_nc)
            Dqo = nqo.dissipator(c_ops_qt)
            Dqt = nqo.dissipator(c_ops_qt, lindblad=True)
            assert np.allclose(Dqo.full(), Dqt.full())
            assert np.allclose(Dnc, Dqt.full())


def test_liovillian():
    for coupling in [0.005, 0.05, 0.5]:
        H_parameters = Eg, delta, omegac, coupling
        Hqt0, Hqt1, _ = nqtls.Hamiltonian(*H_parameters)
        Hnc0, Hnc1, _ = ntls.Hamiltonian(*H_parameters)
        Hqt = Hqt0 + Hqt1
        Hnc = Hnc0 + Hnc1
        for VL, VR in [[-3, 3], [-1, 3], [0, 3], [1, 3]]:
            c_ops_qt = list(
                nqtls.collapses(H_parameters, VL, VR, kappa, gL, gR, kT)
            )
            Lqt = nqo.liouvillian(Hqt, c_ops_qt)

            c_ops_nc = ntls.collapses(H_parameters, VL, VR, kappa, gL, gR, kT)
            Lnc = no.liouvillian(Hnc, c_ops_nc)
            assert np.allclose(Lnc, Lqt.full())
