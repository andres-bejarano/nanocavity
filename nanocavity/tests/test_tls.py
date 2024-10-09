import qutip as qt
import numpy as np
import nanocavity.master_equation as nme
import nanocavity.tls as ntls
import nanocavity.qutip.tls as nqtls
import nanocavity.operators as no
import nanocavity.jaynes_cumming_analytics as jc


# system parameters
Eg = 0.4
Delta = 0.9
hw_ph = 1
g_ph = 0.5
U = 1

H_parameters = Eg, Delta, hw_ph, g_ph, U

# bath parameters
Gamma_L, Gamma_R = 1e-3, 2e-3
kappa = 0.1
kT = 0.1


# the angle of each branch in Jaynes-Cummings model
# \ket{+}_n = \cos{\theta_n}\ket{ng}-i \sin{\theta_n}\ket{n-1,e}
# \ket{-}_n = \sin{\theta_n}\ket{ng}+i \cos{\theta_n}\ket{n-1,e}
def theta(n):
    delta = hw_ph - Delta
    theta = 0.5 * np.arctan(2 * np.sqrt(n) * g_ph / delta)
    return theta


def A_theta():
    tan_theta2 = np.tan(theta(1)) ** 2
    return tan_theta2 + 1 / tan_theta2


# formulas in issue #138
def analytics_sec():
    # huge kappa to have better precision in our analytics
    kappa = 1

    x = Gamma_L / Gamma_R
    y = (Gamma_L + Gamma_R) / (2 * kappa)
    tan2 = np.tan(theta(1)) ** 2

    Pg = 2 / (2 + x + 1 / x + A_theta() * y)

    Pplus = tan2 * y * Pg
    Pminus = (1 / tan2) * y * Pg

    P0 = (1 / (2 * x)) * Pg
    Pge = (x / 2) * Pg

    return Pg, np.array([P0, Pminus, Pplus, Pge])


# peding to add populations coming from nanocavity.rate_equation()
def test_populations():
    kT = 1e-2
    kappa = 1
    VL, VR = 10, -10
    max_bosons = 1

    H_parameters = Eg, Delta, hw_ph, g_ph, U, max_bosons
    H0, Hint, [dg, de, a] = ntls.Hamiltonian(*H_parameters)
    H = H0 + Hint
    c_ops = ntls.collapses(H, [dg, de, a], VL, VR, kappa, Gamma_L, Gamma_R, kT)
    L = no.liouvillian(H, c_ops)
    rho = nme.stationary(L)
    # rho is written in the basis without interaction
    _, V = H.eigh()
    Vinv = np.linalg.inv(V)
    rho = Vinv @ rho @ V
    Pme = rho.diagonal().real

    Pgme, Pme = Pme[1], [Pme[0], Pme[2], Pme[4], Pme[6]]
    Pgan, Pan = analytics_sec()
    assert np.allclose(Pgme, Pgan, atol=1e-2)
    assert np.allclose(Pme, Pan, atol=1e-3)


def test_stationary():
    g_ph = 0.05
    for VL, VR in [[3, -3], [2, 0], [-1, 2]]:
        max_bosons = 1
        H0, Hint, [dg, de, a] = ntls.Hamiltonian(Eg, Delta, hw_ph, g_ph, U, max_bosons)
        H = H0 + Hint

        c_ops = ntls.collapses(H, [dg, de, a], VL, VR, kappa, Gamma_L, Gamma_R, kT)
        L = no.liouvillian(H, c_ops)
        Pme = nme.stationary(L)

        H0, Hint, [dg, de, a] = nqtls.Hamiltonian(Eg, Delta, hw_ph, g_ph, U, max_bosons)
        c_ops = nqtls.collapses(
            H0 + Hint, [dg, de, a], VL, VR, kappa, Gamma_L, Gamma_R, kT
        )
        Pqt = qt.steadystate(H0 + Hint, c_ops).full()
        assert np.allclose(Pme, Pqt)


def test_correlation_AB():
    tlist = np.linspace(0.0, 200, 100)
    g_ph = 0.05
    max_bosons = 1
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
        c_nc = ntls.collapses(
            Hnc, [dg_nc, de_nc, a_nc], VL, VR, kappa, Gamma_L, Gamma_R, kT
        )
        L = no.liouvillian(Hnc0 + Hnc1, c_nc)
        Snc = nme.correlation_AB(a_nc.d, L, a_nc, tlist)

        c_qt = nqtls.collapses(
            Hqt, [dg_qt, de_qt, a_qt], VL, VR, kappa, Gamma_L, Gamma_R, kT
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


def test_spectrum_analytics():
    kT = 1e-2
    wlist = np.linspace(0.0, 1.8, 10)
    VL, VR = 10, -10
    max_bosons = 1
    H_parameters = Eg, Delta, hw_ph, g_ph, U, max_bosons, True
    H0, Hint, [dg, de, a] = ntls.Hamiltonian(*H_parameters)
    H = H0 + Hint
    for iva in [False, True]:
        if iva:
            H = H0
        else:
            H = H0 + Hint
        c_ops = ntls.collapses(H, [dg, de, a], VL, VR, kappa, Gamma_L, Gamma_R, kT)
        L = no.liouvillian(H0 + Hint, c_ops)
        Inc = kappa * nme.spectrum(L, a, wlist)
        Ianalytics = jc.spectrum(
            H_parameters, VL, VR, kappa, Gamma_L, Gamma_R, kT, wlist, iva=iva
        )
        assert np.allclose(Inc, Ianalytics, atol=1e-6)


def test_spectrum_g2():
    wlist = np.linspace(0.0, 1.8, 13)
    tlist = np.linspace(0.0, 300, 10)

    VL, VR = 10, -10

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
        Hnc, [dg_nc, de_nc, a_nc], VL, VR, kappa, Gamma_L, Gamma_R, kT, total=False
    )
    [c_gpL, c_epL, c_gpR, c_epR, c_ap] = c_nc_p
    [c_gmL, c_emL, c_gmR, c_emR, c_am] = c_nc_m
    c_nc = c_gpL + c_epL + c_gpR + c_epR + c_ap + c_gmL + c_emL + c_gmR + c_emR + c_am
    L = no.liouvillian(Hnc, c_nc)
    Inc = kappa * nme.spectrum(L, a_nc, wlist)
    _, c_am = no.collapses(a_nc, Hnc, kT, bath="bosonic", rate=kappa, total=False)
    J = no.jump(c_am)
    g2nc = nme.g2(L, J, tlist)

    c_qt = nqtls.collapses(
        Hqt, [dg_qt, de_qt, a_qt], VL, VR, kappa, Gamma_L, Gamma_R, kT
    )
    Iqt = kappa / (2 * np.pi) * qt.spectrum(Hqt, wlist, c_qt, a_qt.dag(), a_qt)
    rho_st = qt.steadystate(Hqt, list(c_qt))
    g2qt, _ = qt.coherence_function_g2(
        Hqt, state0=rho_st, taulist=tlist, c_ops=c_qt, a_op=a_qt, solver="me"
    )

    assert np.allclose(Inc, Iqt)
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
