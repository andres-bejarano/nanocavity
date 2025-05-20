import numpy as np
import pytest
import qutip as qt

import nanocavity.master_equation as nme
import nanocavity.operators as no
import nanocavity.qutip.operators as nqo
import nanocavity.qutip.tls as nqtls
import nanocavity.tls as ntls


def test_Htls_nc_QuTiP():
    for rwa in (True, False):
        for max_bosons in (1, 2):
            Eg = 0.4
            Delta = 0.9
            hw_ph = 1
            g_ph = 0.3
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
    Eg = 0.4
    Delta = 0.9
    hw_ph = 1
    g_ph = 0.05
    U = 0
    kappa = 0.1
    Gamma_L = 1e-3
    Gamma_R = 2e-3

    VL = 3
    VR = -3
    kT = 0.1
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
        Cqtp, Cqtm = nqtls.collapses(
            Hqt,
            [dg_qt, de_qt, a_qt],
            VL,
            VR,
            kappa,
            Gamma_L,
            Gamma_R,
            kT,
            hw_ph,
            total=False,
        )
        Cqtp = [c for sub in Cqtp for c in sub]
        Cqtm = [c for sub in Cqtm for c in sub]
        Cncp, Cncm = ntls.collapses(
            Hnc, [dg_nc, de_nc, a_nc], VL, VR, kappa, Gamma_L, Gamma_R, kT, hw_ph
        )
        Cqt = []
        Cnc = []
        for c in Cqtp:
            Cqt.append(c.full())
        for coll in Cncp:
            for k in coll:
                for c in coll[k]:
                    Cnc.append(c)
        for c in Cqtm:
            Cqt.append(c.full())
        for coll in Cncm:
            for k in coll:
                for c in coll[k]:
                    Cnc.append(c)
        Cqt = np.array(Cqt).flatten()
        Cnc = np.array(Cnc).flatten()
        Cqt.sort()
        Cnc.sort()

        assert np.allclose(Cnc, Cqt)


def test_jump_operator():
    Eg = 0.4
    Delta = 0.9
    hw_ph = 1
    g_ph = 0.3

    U = 0
    kappa = 0.1
    Gamma_L = 1e-3
    Gamma_R = 2e-3

    VL = 3
    VR = -3
    kT = 0.1
    max_bosons = 1
    H_parameters = Eg, Delta, hw_ph, g_ph, U, max_bosons
    Hnc0, Hnc1, [dg_nc, de_nc, a_nc] = ntls.Hamiltonian(*H_parameters)
    Hqt0, Hqt1, [dg_qt, de_qt, a_qt] = nqtls.Hamiltonian(*H_parameters)
    Hnc = Hnc0 + Hnc1
    Hqt = Hqt0 + Hqt1
    for VL, VR in [[-3, 3], [-1, 3], [0, 3], [1, 3]]:
        c_qt = nqtls.collapses(
            Hqt, [dg_qt, de_qt, a_qt], VL, VR, kappa, Gamma_L, Gamma_R, kT, hw_ph
        )
        cp_nc, cm_nc = ntls.collapses(
            Hnc, [dg_nc, de_nc, a_nc], VL, VR, kappa, Gamma_L, Gamma_R, kT, hw_ph
        )
        c_nc = cp_nc + cm_nc
        c_nc = [c for sub in c_nc for c in sub]

        Jqt = nqo.jump(c_qt)
        Jnc = no.jump(c_nc)

        assert np.allclose(Jqt.full(), Jnc)


@pytest.mark.slow
def test_dissipators():
    g_ph = 0.05
    Eg = 0.4
    Delta = 0.9
    hw_ph = 1
    g_ph = 0.3

    U = 0
    kappa = 0.1
    Gamma_L = 1e-3
    Gamma_R = 2e-3

    VL = 3
    VR = -3
    kT = 0.1
    max_bosons = 1
    for VL, VR in [[-3, 3], [-1, 3], [0, 3], [1, 3]]:
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
                Hqt, [dg_qt, de_qt, a_qt], VL, VR, kappa, Gamma_L, Gamma_R, kT, hw_ph
            )
        )
        c_opsp_nc, c_opsm_nc = ntls.collapses(
            Hnc, [dg_nc, de_nc, a_nc], VL, VR, kappa, Gamma_L, Gamma_R, kT, hw_ph
        )

        Dnc = 0
        for c in c_opsp_nc:
            Dnc += no.dissipator(c)
        for c in c_opsm_nc:
            Dnc += no.dissipator(c)

        Dqo = nqo.dissipator(c_ops_qt)
        Dqt = nqo.dissipator(c_ops_qt, lindblad=True)
        assert np.allclose(Dqo.full(), Dqt.full())
        assert np.allclose(Dnc, Dqt.full())


def test_liouvillian():
    Eg = 0.4
    Delta = 0.9
    hw_ph = 1

    U = 0
    kappa = 0.1
    Gamma_L = 1e-3
    Gamma_R = 2e-3

    VL = 3
    VR = -3
    kT = 0.1
    max_bosons = 1
    g_ph = 0.05
    Hqt0, Hqt1, [dg_qt, de_qt, a_qt] = nqtls.Hamiltonian(
        Eg, Delta, hw_ph, g_ph, U, max_bosons
    )
    Hnc0, Hnc1, [dg_nc, de_nc, a_nc] = ntls.Hamiltonian(
        Eg, Delta, hw_ph, g_ph, U, max_bosons
    )
    Hqt = Hqt0 + Hqt1
    Hnc = Hnc0 + Hnc1
    for VL, VR in [[-3, 3], [0, 3], [1, 3]]:
        c_ops_qt = list(
            nqtls.collapses(
                Hqt, [dg_qt, de_qt, a_qt], VL, VR, kappa, Gamma_L, Gamma_R, kT, hw_ph
            )
        )
        Lqt = nqo.liouvillian(Hqt, c_ops_qt)

        c_opsp_nc, c_opsm_nc = ntls.collapses(
            Hnc, [dg_nc, de_nc, a_nc], VL, VR, kappa, Gamma_L, Gamma_R, kT, hw_ph
        )
        c_ops_nc = c_opsp_nc + c_opsm_nc
        c_ops_nc = [c for sub in c_ops_nc for c in sub]
        Lnc = no.liouvillian(Hnc, c_ops_nc)
        assert np.allclose(Lnc, Lqt.full())


def test_stationary():
    Eg = 0.4
    Delta = 0.9
    hw_ph = 1
    U = 1
    g_ph = 0.05
    Gamma_L, Gamma_R = 1e-3, 2e-3
    kappa = 0.1
    kT = 0.1
    for VL, VR in [[3, -3], [2, 0], [-1, 2]]:
        max_bosons = 1
        H0, Hint, [dg, de, a] = ntls.Hamiltonian(Eg, Delta, hw_ph, g_ph, U, max_bosons)
        H = H0 + Hint

        c_opsp, c_opsm = ntls.collapses(
            H, [dg, de, a], VL, VR, kappa, Gamma_L, Gamma_R, kT, hw_ph
        )
        c_ops = c_opsp + c_opsm
        c_ops = [c for sub in c_ops for c in sub]
        L = no.liouvillian(H, c_ops)
        Pme = nme.stationary(L)

        H0, Hint, [dg, de, a] = nqtls.Hamiltonian(Eg, Delta, hw_ph, g_ph, U, max_bosons)
        c_ops = nqtls.collapses(
            H0 + Hint, [dg, de, a], VL, VR, kappa, Gamma_L, Gamma_R, kT, hw_ph
        )
        Pqt = qt.steadystate(H0 + Hint, c_ops).full()
        assert np.allclose(Pme, Pqt)


@pytest.mark.slow
def test_correlation_AB():
    tlist = np.linspace(0.0, 200, 100)
    Eg = 0.4
    Delta = 0.9
    hw_ph = 1
    U = 1
    g_ph = 0.05
    max_bosons = 1
    Gamma_L, Gamma_R = 1e-3, 2e-3
    kappa = 0.1
    kT = 0.1
    Hnc0, Hnc1, [dg_nc, de_nc, a_nc] = ntls.Hamiltonian(
        Eg, Delta, hw_ph, g_ph, U, max_bosons
    )
    Hqt0, Hqt1, [dg_qt, de_qt, a_qt] = nqtls.Hamiltonian(
        Eg, Delta, hw_ph, g_ph, U, max_bosons
    )
    VL, VR = 3, -3
    for iva in (False, True):
        if iva:
            Hnc = Hnc0
            Hqt = Hqt0
        else:
            Hnc = Hnc0 + Hnc1
            Hqt = Hqt0 + Hqt1
        c_ncp, c_ncm = ntls.collapses(
            Hnc, [dg_nc, de_nc, a_nc], VL, VR, kappa, Gamma_L, Gamma_R, kT, hw_ph
        )
        c_nc = c_ncp + c_ncm
        c_nc = [c for sub in c_nc for c in sub]
        L = no.liouvillian(Hnc0 + Hnc1, c_nc)
        Snc = nme.correlation_AB(a_nc.d, L, a_nc, tlist)

        c_qt = nqtls.collapses(
            Hqt, [dg_qt, de_qt, a_qt], VL, VR, kappa, Gamma_L, Gamma_R, kT, hw_ph
        )

        rho_st = qt.steadystate(Hqt0 + Hqt1, list(c_qt))
        Sqt = qt.correlation_2op_1t(
            H=Hqt0 + Hqt1,
            state0=rho_st,
            taulist=tlist,
            c_ops=list(c_qt),
            a_op=a_qt.dag(),
            b_op=a_qt,
        )
        assert np.allclose(Snc.real, Sqt.real, atol=1e-5)
        assert np.allclose(Snc.imag, Sqt.imag, atol=1e-5)


@pytest.mark.slow
def test_spectrum_g2():
    wlist = np.linspace(0.0, 1.8, 13)
    tlist = np.linspace(0.0, 300, 10)

    VL, VR = 10, -10

    Eg = 0.4
    hw_ph = 1
    U = 1
    Gamma_L, Gamma_R = 1e-6, 2e-6
    kappa = 0.1
    kT = 0.1

    # g2 only matches at this values
    g_ph = 0.005
    Delta = 0.99
    max_bosons = 1

    H_parameters = Eg, Delta, hw_ph, g_ph, U, max_bosons
    Hnc0, Hnc1, [dg_nc, de_nc, a_nc] = ntls.Hamiltonian(*H_parameters)
    Hnc = Hnc0 + Hnc1

    Hqt0, Hqt1, [dg_qt, de_qt, a_qt] = nqtls.Hamiltonian(*H_parameters)
    Hqt = Hqt0 + Hqt1

    c_nc_p, c_nc_m = ntls.collapses(
        Hnc,
        [dg_nc, de_nc, a_nc],
        VL,
        VR,
        kappa,
        Gamma_L,
        Gamma_R,
        kT,
        hw_ph,
    )
    [c_gpL, c_epL, c_gpR, c_epR, c_ap] = c_nc_p
    [c_gmL, c_emL, c_gmR, c_emR, c_am] = c_nc_m
    c_nc = c_gpL + c_epL + c_gpR + c_epR + c_ap + c_gmL + c_emL + c_gmR + c_emR + c_am
    L = no.liouvillian(Hnc, c_nc)
    Inc = kappa * nme.spectrum(L, a_nc, wlist)
    basis = Hnc.eigh()
    _, c_am = no.collapses(a_nc, basis, kT, bath="bosonic", rate=kappa)
    g2nc = nme.g2(L, a_nc, tlist)

    c_qt = nqtls.collapses(
        Hqt, [dg_qt, de_qt, a_qt], VL, VR, kappa, Gamma_L, Gamma_R, kT, hw_ph
    )
    Iqt = kappa / (2 * np.pi) * qt.spectrum(Hqt, wlist, c_qt, a_qt.dag(), a_qt)
    rho_st = qt.steadystate(Hqt, list(c_qt))
    g2qt, _ = qt.coherence_function_g2(
        Hqt, state0=rho_st, taulist=tlist, c_ops=c_qt, a_op=a_qt, solver="me"
    )

    assert np.allclose(Inc, Iqt, atol=1e-5)
    assert np.allclose(g2nc, g2qt, atol=1e-1)


"""peding to develop
def test_current():
    for VL, VR in [[3, -3], [2, 0], [-1, 2]]:
            [dg, de, a], Hqt, c_ops = tls.collapses('qutip', H_parameters, VL, VR, kappa, gL, gR, kT, alone=False)

            #left electrode
            cp_gL, cm_gL = qo.collapses( dg, Hqt, kT, bath='fermionic', mu=VL, total=False)
            cp_eL, cm_eL = qo.collapses(de, Hqt, kT, bath='fermionic', mu=VL, total=False)
            CpL = list(np.sqrt(gL) * np.array(cp_gL + cp_eL))
            CmL = list(np.sqrt(gL) * np.array(cm_gL + cm_eL))

            #cavity mode
            cap, cam = qo.collapses(a, Hqt, kT, bath='bosonic', total=False)
            Cp = list(np.sqrt(kappa) * np.array(cap))
            Cm = list(np.sqrt(kappa) * np.array(cam))

            L = qo.liouvillian(Hqt, list(c_ops))

            Ig_me = qme.current(qo.jump(Cm) - qo.jump(Cp), L)
            Ie_me = qme.current(qo.jump(CpL) - qo.jump(CmL), L)
            Ig_re, Ie_re, _, _ = fcs(VL, VR)

            #assert np.allclose(Ig_me, Ig_re)
            #assert np.allclose(Ie_me, Ie_re)"""
