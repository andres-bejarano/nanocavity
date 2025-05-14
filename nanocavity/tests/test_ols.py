import nanocavity.ols as nols
import nanocavity.operators as no
import nanocavity.master_equation as nme
import nanocavity.franck_condon as nfc
import nanocavity.distributions as nd
from itertools import chain
from secondquant.operator import Operator
import numpy as np
import secondquant as sq


def test_hamiltonian():
    Hs, anni_ops, num_ops = nols.Hamiltonian(1)
    assert isinstance(Hs, Operator)
    assert isinstance(anni_ops, list)
    assert isinstance(anni_ops[0], Operator)
    assert isinstance(anni_ops[1], Operator)
    assert isinstance(num_ops, list)
    assert isinstance(num_ops[0], Operator)
    assert isinstance(num_ops[1], Operator)


def test_lang_firsov_transform():
    hw_ph = 1
    g_ph = 1
    Hs, [D, a_ph], [ng, n_ph] = nols.Hamiltonian(hw_ph)
    D, A = nols.Lang_Firsov_transform(D, a_ph, g_ph)
    assert isinstance(D, Operator)
    assert isinstance(A, Operator)


def test_collapse_electronic():
    hw_ph = 1
    g_ph = 1
    Hs, [D, a_ph], [ng, n_ph] = nols.Hamiltonian(hw_ph)
    D, A = nols.Lang_Firsov_transform(D, a_ph, g_ph)
    basis = Hs.eigh()

    coll_list = nols.collapse_electronic(D, basis, 1, -1, 1e-4, 5e-4, 1e-3)
    assert isinstance(coll_list[0], list)
    assert isinstance(coll_list, tuple)
    assert len(coll_list) == 2  # Plus, Minus


def test_liouvillian():
    hw_ph = 1
    Hs, [dg, a_ph], [ng, n_ph] = nols.Hamiltonian(hw_ph, max_bosons=2)

    Gamma_s = 5e-4
    Gamma_t = 1e-4
    kappa = 0.1
    kT = 1e-3
    g_ph = 0.5
    Vs = 1.5
    Vt = -0.5

    L = nols.liouvillian(dg, a_ph, Hs, g_ph, Vs, Vt, Gamma_s, Gamma_t, kappa, kT)

    assert np.allclose(L.shape[0], Hs.toarray().shape[0] ** 2)
    assert np.allclose(L.shape[1], Hs.toarray().shape[1] ** 2)
    assert np.allclose(L[0, 7], kappa)


def P0_Q0_st(VL, VR, GL, GR, g_ph, hw_ph):
    kT = 0
    Gp00 = G_p_nm_a(0, 0, g_ph, VL, GL, kT) + G_p_nm_a(0, 0, g_ph, VR, GR, kT)
    Gp10 = G_p_nm_a(1, 0, g_ph, VL, GL, kT) + G_p_nm_a(1, 0, g_ph, VR, GR, kT)
    Gm00 = G_m_nm_a(0, 0, g_ph, VL, GL, kT) + G_m_nm_a(0, 0, g_ph, VR, GR, kT)
    Gm10 = G_m_nm_a(1, 0, g_ph, VL, GL, kT) + G_m_nm_a(1, 0, g_ph, VR, GR, kT)
    P0 = Gm00 + Gm10
    P0 /= Gp00 + Gp10 + Gm00 + Gm10
    Q0 = Gp00 + Gp10
    Q0 /= Gp00 + Gp10 + Gm00 + Gm10
    return P0, Q0


def P1_Q1_st(VL, VR, GL, GR, g_ph, hw_ph, kappa):
    kT = 0
    Gp10 = G_p_nm_a(1, 0, g_ph, VL, GL, kT) + G_p_nm_a(1, 0, g_ph, VR, GR, kT)
    Gp11 = G_p_nm_a(1, 1, g_ph, VL, GL, kT) + G_p_nm_a(1, 1, g_ph, VR, GR, kT)
    Gp21 = G_p_nm_a(2, 1, g_ph, VL, GL, kT) + G_p_nm_a(2, 1, g_ph, VR, GR, kT)
    Gm10 = G_m_nm_a(1, 0, g_ph, VL, GL, kT) + G_m_nm_a(1, 0, g_ph, VR, GR, kT)
    Gm11 = G_m_nm_a(1, 1, g_ph, VL, GL, kT) + G_m_nm_a(1, 1, g_ph, VR, GR, kT)
    Gm21 = G_m_nm_a(2, 1, g_ph, VL, GL, kT) + G_m_nm_a(2, 1, g_ph, VR, GR, kT)
    P0, Q0 = P0_Q0_st(VL, VR, GL, VR, g_ph, hw_ph)

    P1 = Gm10 / kappa * Q0 + (Gm11 + Gm21) * Gp10 * P0 / kappa**2
    Q1 = Gp10 / kappa * P0 + (Gp11 + Gp21) * Gm10 * Q0 / kappa**2
    return P1, Q1


def P2_Q2_st(VL, VR, GL, GR, g_ph, hw_ph, kappa):
    kT = 0
    P0, Q0 = P0_Q0_st(VL, VR, GL, VR, g_ph, hw_ph)
    Gp00 = G_p_nm_a(0, 0, g_ph, VL, GL, kT) + G_p_nm_a(0, 0, g_ph, VR, GR, kT)
    Gp10 = G_p_nm_a(1, 0, g_ph, VL, GL, kT) + G_p_nm_a(1, 0, g_ph, VR, GR, kT)
    Gp11 = G_p_nm_a(1, 1, g_ph, VL, GL, kT) + G_p_nm_a(1, 1, g_ph, VR, GR, kT)
    Gp21 = G_p_nm_a(2, 1, g_ph, VL, GL, kT) + G_p_nm_a(2, 1, g_ph, VR, GR, kT)
    Gp22 = G_p_nm_a(2, 2, g_ph, VL, GL, kT) + G_p_nm_a(2, 2, g_ph, VR, GR, kT)
    Gm00 = G_m_nm_a(0, 0, g_ph, VL, GL, kT) + G_m_nm_a(0, 0, g_ph, VR, GR, kT)
    Gm10 = G_m_nm_a(1, 0, g_ph, VL, GL, kT) + G_m_nm_a(1, 0, g_ph, VR, GR, kT)
    Gm11 = G_m_nm_a(1, 1, g_ph, VL, GL, kT) + G_m_nm_a(1, 1, g_ph, VR, GR, kT)
    Gm21 = G_m_nm_a(2, 1, g_ph, VL, GL, kT) + G_m_nm_a(2, 1, g_ph, VR, GR, kT)
    Gm22 = G_m_nm_a(2, 2, g_ph, VL, GL, kT) + G_m_nm_a(2, 2, g_ph, VR, GR, kT)

    P2_f = Gm21 * Gp10 * P0 / (2 * kappa**2)
    P2_s = ((Gm21 * (Gp11 + Gp21) * Gm10 + 0.5 * Gm22 * Gp21 * Gm10) * Q0) / (
        2 * kappa**3
    )
    P2 = P2_f + P2_s

    Q2_f = Gp21 * Gm10 * Q0 / (2 * kappa**2)
    Q2_s = ((Gp21 * (Gm11 + Gm21) * Gp10 + 0.5 * Gp22 * Gm21 * Gp10) * P0) / (
        2 * kappa**3
    )
    Q2 = Q2_f + Q2_s
    return P2, Q2


def G_p_nm_a(n, m, g_ph, V, Gamma, kT):
    Fnm = nfc.FC(n, m, g_ph) ** 2
    f = nd.fermi_dirac(n - m, kT, V)
    return Gamma * f * Fnm


def G_m_nm_a(n, m, g_ph, V, Gamma, kT):
    Fnm = nfc.FC(n, m, g_ph) ** 2
    f = nd.fermi_dirac(m - n, kT, V)
    return Gamma * (1 - f) * Fnm


def test_populations():
    hw_ph = 1
    nmax = 6
    Hs, [dg, a_ph], [ng, n_ph] = nols.Hamiltonian(hw_ph, max_bosons=nmax)

    kappa = 0.1
    kT = 0.001

    VL = 1.1
    VR = -0.1
    GL = 5e-4
    GR = 1e-4
    Gamma = GL + GR

    g_ph = 0.4
    L = nols.liouvillian(dg, a_ph, Hs, g_ph, VL, VR, GL, GR, kappa, kT)
    rho_st = np.real(nme.stationary(L))
    P0 = nme.reduced_population([ng, n_ph], rho_st, [0, 0])
    P1 = nme.reduced_population([ng, n_ph], rho_st, [0, 1])
    P2 = nme.reduced_population([ng, n_ph], rho_st, [0, 2])
    Q0 = nme.reduced_population([ng, n_ph], rho_st, [1, 0])
    Q1 = nme.reduced_population([ng, n_ph], rho_st, [1, 1])
    Q2 = nme.reduced_population([ng, n_ph], rho_st, [1, 2])
    P0_an, Q0_an = P0_Q0_st(VL, VR, GL, GR, g_ph, hw_ph)
    P1_an, Q1_an = P1_Q1_st(VL, VR, GL, GR, g_ph, hw_ph, kappa)
    P2_an, Q2_an = P2_Q2_st(VL, VR, GL, GR, g_ph, hw_ph, kappa)

    assert np.isclose(P0_an, P0, atol=(Gamma / kappa))
    assert np.isclose(Q0_an, Q0, atol=(Gamma / kappa))
    assert np.isclose(P1_an, P1, atol=(Gamma / kappa) ** 2)
    assert np.isclose(Q1_an, Q1, atol=100 * (Gamma / kappa) ** 2)
    assert np.isclose(P2_an, P2, atol=(Gamma / kappa) ** 3)
    assert np.isclose(Q2_an, Q2, atol=(Gamma / kappa) ** 3)
