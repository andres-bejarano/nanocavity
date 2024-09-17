import numpy as np
import nanocavity.operators as no
import nanocavity.qutip.operators as qo
import nanocavity.rate_equation as nre
import nanocavity.tls as tls


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
            Hnc, [Dg, De, A] = tls.Hamiltonian("nanocavity", *H_parameters)
            Hqt, [dg, de, a] = tls.Hamiltonian("qutip", *H_parameters)

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
            Cqt = tls.collapses("qutip", H_parameters, VL, VR, kappa, gL, gR, kT)
            Cnc = tls.collapses("nanocavity", H_parameters, VL, VR, kappa, gL, gR, kT)

            for i in range(len(Cnc)):
                assert np.allclose(Cnc[i], Cqt[i].full())


def test_jump_operator():
    for VL, VR in [[-3, 3], [-1, 3], [0, 3], [1, 3]]:
        c_qt = tls.collapses("qutip", H_parameters, VL, VR, kappa, gL, gR, kT)
        c_nc = tls.collapses("nanocavity", H_parameters, VL, VR, kappa, gL, gR, kT)

        Jqt = qo.jump(c_qt)
        Jnc = no.jump(c_nc)

        assert np.allclose(Jqt.full(), Jnc)


def test_dissipators():
    for coupling in [0.005, 0.05, 0.5]:
        for VL, VR in [[-3, 3], [-1, 3], [0, 3], [1, 3]]:
            H_parameters = Eg, delta, omegac, coupling
            c_ops_qt = list(
                tls.collapses("qutip", H_parameters, VL, VR, kappa, gL, gR, kT)
            )
            c_ops_nc = list(
                tls.collapses("nanocavity", H_parameters, VL, VR, kappa, gL, gR, kT)
            )

            Dnc = no.dissipator(c_ops_nc)
            Dqo = qo.dissipator(c_ops_qt)
            Dqt = qo.dissipator(c_ops_qt, lindblad=True)
            assert np.allclose(Dqo.full(), Dqt.full())
            assert np.allclose(Dnc, Dqt.full())


def test_liovillian():
    for coupling in [0.005, 0.05, 0.5]:
        H_parameters = Eg, delta, omegac, coupling
        Hqt, _ = tls.Hamiltonian("qutip", *H_parameters)
        Hnc, _ = tls.Hamiltonian("nanocavity", *H_parameters)
        for VL, VR in [[-3, 3], [-1, 3], [0, 3], [1, 3]]:
            c_ops_qt = list(
                tls.collapses("qutip", H_parameters, VL, VR, kappa, gL, gR, kT)
            )
            Lqt = qo.liouvillian(Hqt, c_ops_qt)

            c_ops_nc = list(
                tls.collapses("nanocavity", H_parameters, VL, VR, kappa, gL, gR, kT)
            )
            Lnc = no.liouvillian(Hnc, c_ops_nc)
            assert np.allclose(Lnc, Lqt.full())
