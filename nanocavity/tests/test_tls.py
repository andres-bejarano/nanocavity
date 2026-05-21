import numpy as np
import pytest

import nanocavity.jaynes_cumming_analytics as jc
import nanocavity.master_equation as nme
import nanocavity.operators as no
import nanocavity.tls as ntls


# the angle of each branch in Jaynes-Cummings model
# \ket{+}_n = \cos{\theta_n}\ket{ng}-i \sin{\theta_n}\ket{n-1,e}
# \ket{-}_n = \sin{\theta_n}\ket{ng}+i \cos{\theta_n}\ket{n-1,e}
def theta(n, hw_ph, Delta, g_ph):
    delta = hw_ph - Delta
    theta = 0.5 * np.arctan(2 * np.sqrt(n) * g_ph / delta)
    return theta


def A_theta(hw_ph, Delta, g_ph):
    tan_theta2 = np.tan(theta(1, hw_ph, Delta, g_ph)) ** 2
    return tan_theta2 + 1 / tan_theta2


# formulas in issue #138
def analytics_sec(Gamma_L, Gamma_R, hw_ph, Delta, g_ph):
    # huge kappa to have better precision in our analytics
    kappa = 1

    x = Gamma_L / Gamma_R
    y = (Gamma_L + Gamma_R) / (2 * kappa)
    tan2 = np.tan(theta(1, hw_ph, Delta, g_ph)) ** 2

    Pg = 2 / (2 + x + 1 / x + A_theta(hw_ph, Delta, g_ph) * y)

    Pplus = tan2 * y * Pg
    Pminus = (1 / tan2) * y * Pg

    P0 = (1 / (2 * x)) * Pg
    Pge = (x / 2) * Pg

    return Pg, np.array([P0, Pminus, Pplus, Pge])


# peding to add populations coming from nanocavity.rate_equation()
def test_populations():
    Gamma_L, Gamma_R = 1e-6, 2e-6
    Eg = 0.4
    Delta = 0.9
    hw_ph = 1
    U = 1
    g_ph = 5e-2
    kT = 1e-2
    kappa = 0.01
    VL, VR = 10, -10
    max_bosons = 1

    H_parameters = Eg, Delta, hw_ph, g_ph, U, max_bosons
    H0, Hint, [dg, de, a] = ntls.Hamiltonian(*H_parameters)
    H = H0 + Hint
    c_opsp, c_opsm = ntls.collapses(
        H, [dg, de, a], VL, VR, kappa, Gamma_L, Gamma_R, kT, hw_ph
    )
    c_ops = c_opsp + c_opsm
    L = no.liouvillian(H, c_ops)
    rho = nme.stationary(L)
    # rho is written in the basis without interaction
    _, V = H.eigh()
    Vinv = np.linalg.inv(V)
    rho = Vinv @ rho @ V
    Pme = rho.diagonal().real

    Pgme, Pme = Pme[1], [Pme[0], Pme[2], Pme[4], Pme[6]]
    Pgan, Pan = analytics_sec(Gamma_L, Gamma_R, hw_ph, Delta, g_ph)
    assert np.allclose(Pgme, Pgan, atol=1e-2)
    assert np.allclose(Pme, Pan, atol=1e-3)


@pytest.mark.slow
def test_spectrum_analytics():
    Eg = 0.4
    Delta = 0.9
    hw_ph = 1
    U = 1
    g_ph = 0.5

    Gamma_L, Gamma_R = 1e-5, 2e-5
    kappa = 0.1
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
        c_opsp, c_opsm = ntls.collapses(
            H, [dg, de, a], VL, VR, kappa, Gamma_L, Gamma_R, kT, hw_ph
        )
        c_ops = c_opsp + c_opsm
        L = no.liouvillian(H0 + Hint, c_ops)
        Inc = kappa * nme.spectrum(L, a, wlist)
        Ianalytics = jc.spectrum(
            H_parameters, VL, VR, kappa, Gamma_L, Gamma_R, kT, wlist, iva=iva
        )
        assert np.allclose(Inc, Ianalytics, atol=1e-6)


def test_spectrum_with_and_without_iva():
    # in the strong-coupling regime the two approaches should coincide
    Eg = -0.4
    Delta = 0.7
    hw_ph = 1.0
    g_ph = 0.2
    U = 2
    H0, Hint, [dg, de, a] = ntls.Hamiltonian(
        Eg=Eg, Delta=Delta, hw_ph=hw_ph, g_ph=g_ph, U=U, max_bosons=2, rwa=True
    )

    kappa = 0.05
    VL = Eg - 0.5
    VR = Eg + 1.5  # sufficiently high voltage to populate both P+ and P-
    Gamma_L = Gamma_R = 1e-5
    kT = 0.01
    cutoff = 1e-16

    # IVA, collapses in bare basis
    cp_iva, cm_iva = ntls.collapses(
        H0,  # bare H0
        [dg, de, a],
        VL=VL,
        VR=VR,
        kappa=kappa,
        Gamma_L=Gamma_L,
        Gamma_R=Gamma_R,
        kT=kT,
        hw_ph=hw_ph,
    )
    L_iva = no.liouvillian(H0 + Hint, cp_iva + cm_iva)
    rho_iva = nme.stationary(L_iva)
    wlist = np.linspace(0, 2, 21)
    S_iva = nme.spectrum(L_iva, a, wlist, rho_st=rho_iva, cutoff=cutoff)

    # Collapses in dressed basis
    cp, cm = ntls.collapses(
        H0 + Hint,  # full H
        [dg, de, a],
        VL=VL,
        VR=VR,
        kappa=kappa,
        Gamma_L=Gamma_L,
        Gamma_R=Gamma_R,
        kT=kT,
        hw_ph=hw_ph,
    )
    L = no.liouvillian(H0 + Hint, cp + cm)
    rho = nme.stationary(L)
    S = nme.spectrum(L, a, wlist, rho_st=rho, cutoff=cutoff)

    # the tolerances are not super-tight:
    assert np.allclose(rho_iva, rho, atol=1e-5)
    assert np.allclose(S_iva, S, rtol=3e-3, atol=1e-16)


def test_direct_tunneling_term():
    Eg = -0.4
    hw_ph = 1
    H0, Hint, [dg, de, a] = ntls.Hamiltonian(
        Eg=Eg, Delta=0.7, hw_ph=hw_ph, g_ph=0.2, U=2, max_bosons=2, rwa=True
    )

    kappa = 0.05
    VL = Eg - 0.5
    VR = Eg + 1.5  # sufficiently high voltage to populate both P+ and P-
    Gamma_L = Gamma_R = 1e-5
    kT = 0.01
    wlist = np.linspace(0, 2, 21)

    cp0, cm0 = ntls.collapses(
        H0,  # bare H0
        [dg, de, a],
        VL=VL,
        VR=VR,
        kappa=kappa,
        Gamma_L=Gamma_L,
        Gamma_R=Gamma_R,
        kT=kT,
        hw_ph=hw_ph,
    )
    L0 = no.liouvillian(H0 + Hint, cp0 + cm0)
    rho0 = nme.stationary(L0)
    S0 = nme.spectrum(L0, a, wlist, rho_st=rho0)

    gamma_direct = 1e-5
    cp, cm = ntls.collapses(
        H0,  # bare H0
        [dg, de, a],
        VL=VL,
        VR=VR,
        kappa=kappa,
        Gamma_L=Gamma_L,
        Gamma_R=Gamma_R,
        kT=kT,
        hw_ph=hw_ph,
        gamma_direct=gamma_direct,
    )
    L = no.liouvillian(H0 + Hint, cp + cm)
    rho = nme.stationary(L)
    S = nme.spectrum(L, a, wlist, rho_st=rho)

    assert np.all(S > S0)
