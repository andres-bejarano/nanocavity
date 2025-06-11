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

    Gamma_s = 1e-4
    Gamma_t = 5e-4
    kappa = 0.1
    kT = 1e-3
    g_ph = 0.5
    Vs = -0.5
    Vt = 1.5

    L = nols.liouvillian(dg, a_ph, Hs, g_ph, Vs, Vt, Gamma_s, Gamma_t, kappa, kT)

    assert np.allclose(L.shape[0], Hs.toarray().shape[0] ** 2)
    assert np.allclose(L.shape[1], Hs.toarray().shape[1] ** 2)
    assert np.allclose(L[0, 7], kappa)


def test_populations():
    hw_ph = 1
    max_bosons = 5
    Hs, [dg, a_ph], [ng, n_ph] = nols.Hamiltonian(hw_ph, max_bosons=max_bosons)

    kappa = 0.1
    kT = 0.01

    Vt = 0.5
    Vs = -1.5
    Gamma_t = 1e-6
    Gamma_s = 5e-6

    g_ph = 0.5

    for VL, VR in [[1.5, 0.5], [2.5, -0.5], [1.5, -1.5]]:
        L = nols.liouvillian(dg, a_ph, Hs, g_ph, Vs, Vt, Gamma_s, Gamma_t, kappa, kT)
        rho_st = np.real(nme.stationary(L))
        index = nols.get_diagonal_indices(max_bosons=max_bosons, cutoff=2)

        row, colum = zip(*index)
        Pnum = rho_st[list(row), list(colum)]
        Pan = nols.Pst(g_ph, Vt, Vs, Gamma_t, Gamma_s, kappa, kT)
        assert np.allclose(Pnum, Pan)


def test_G_pm_nm_a():
    qi, qf = np.meshgrid([0, 1], [0, 1], indexing="ij")
    qf, qi = np.meshgrid(1, 3, indexing="ij")
    kT = 1e-3
    g_ph = 0.3
    Gp, Gm = nols.G_pm_nm_a(1, 1, 1, 0.5, kT)
    assert np.isclose(Gp, 0)
    assert np.isclose(Gm, 0)
    Gp, Gm = nols.G_pm_nm_a(3, 1, g_ph, 0.5, kT)
    assert np.isclose(Gp, 0)
    assert np.isclose(Gm, 0)

    Gp, Gm = nols.G_pm_nm_a(3, 1, g_ph, 2.5, kT)
    assert np.isclose(Gp, nfc.FC(3, 1, g_ph) ** 2)
    assert np.isclose(Gm, 0)

    # Add these two tests back for Gp and Gm
    Gp, Gm = nols.G_pm_nm_a(1, 3, g_ph, 0.5, kT)
    assert np.isclose(Gp, nfc.FC(1, 3, g_ph) ** 2)
    assert np.isclose(Gm, nfc.FC(1, 3, g_ph) ** 2)

    Gp, Gm = nols.G_pm_nm_a(1, 3, g_ph, 2.5, kT)
    assert np.isclose(Gp, nfc.FC(1, 3, g_ph) ** 2)
    assert np.isclose(Gm, 0)

    Gp, Gm = nols.G_pm_nm_a(1, 3, g_ph, 2.5, kT, DE=0.3)
    assert np.isclose(Gp, nfc.FC(1, 3, g_ph) ** 2)
    assert np.isclose(Gm, 0)

    Gp, Gm = nols.G_pm_nm_a(1, 3, g_ph, 2.5, kT, DE=0.8)
    assert np.isclose(Gp, nfc.FC(1, 3, g_ph) ** 2)
    assert np.isclose(Gm, nfc.FC(1, 3, g_ph) ** 2)

    Gp, Gm = nols.G_pm_nm_a([0, 1], [0, 1], g_ph, 0.5, kT)
    assert Gp.shape == (2, 2)
    assert Gm.shape == (2, 2)
    assert np.isclose(Gp[0, 0], nfc.FC(0, 0, g_ph) ** 2)
    assert np.isclose(Gp[0, 1], nfc.FC(1, 0, g_ph) ** 2)
    assert np.isclose(Gp[1, 0], 0)
    assert np.isclose(Gp[1, 1], nfc.FC(1, 1, g_ph) ** 2)

    assert np.isclose(Gm[0, 0], 0)
    assert np.isclose(Gm[0, 1], nfc.FC(1, 0, g_ph) ** 2)
    assert np.isclose(Gm[1, 0], 0)
    assert np.isclose(Gm[1, 1], 0)

    Gp, Gm = nols.G_pm_nm_a([0, 1], 0, g_ph, 0.5, kT)
    assert Gp.shape == (2,)
    assert Gm.shape == (2,)


def test_get_diagonal_indices():
    Nmax = 5
    index = nols.get_diagonal_indices(max_bosons=Nmax, cutoff=2)

    assert len(index) == 6
    for i in range(3):
        assert index[i] == [i, i]
        assert index[i + 3] == [Nmax + 1 + i, Nmax + 1 + i]


def get_basis_index(n, q, max_bosons):

    max_bosons = 4
    for n in range(max_bosons + 1):
        result = get_basis_index(n, 0, max_bosons)
        expected = n
        assert result == expected

    # Test for q = 1
    for n in range(max_bosons + 1):
        result = get_basis_index(n, 1, max_bosons)
        expected = max_bosons + 1 + n
        assert result == expected

    # Test for invalid charge
    try:
        get_basis_index(0, 2, max_bosons)
        assert False, "Expected ValueError for invalid charge q=2"
    except ValueError:
        pass  # This is the expected behavior


def test_get_vectorized_rho_index():
    # we check population index
    N_ph_max = 4
    index_h, index_e = [], []
    for i in range(3):
        index_h.append(
            nols.get_vectorized_rho_index(n1=i, q1=0, n2=i, q2=0, max_bosons=N_ph_max)
        )
        index_e.append(
            nols.get_vectorized_rho_index(n1=i, q1=1, n2=i, q2=1, max_bosons=N_ph_max)
        )
    index = index_h + index_e

    # index_by_hand
    dim = 2 * (N_ph_max + 1)

    i_h0 = 0
    i_h1 = dim + 1
    i_h2 = i_h1 + dim + 1

    i_e0 = (N_ph_max + 1) * dim + (N_ph_max + 1)
    i_e1 = i_e0 + dim + 1
    i_e2 = i_e1 + dim + 1

    index2 = [i_h0, i_h1, i_h2, i_e0, i_e1, i_e2]

    assert index == index2


def test_Liovillian_matrix_elements():
    hw_ph = 1
    g_ph = 0.5
    N_ph_max = 5

    # we need Gamma<<kappa
    Gamma_s, Gamma_t = 5e-6, 1e-6
    kappa = 0.1
    kT = 0.01

    # first photon emission threshold
    Vs, Vt = 0.5, -1.5
    Hs, [dg, a_ph], [ng, n_ph] = nols.Hamiltonian(hw_ph=hw_ph, max_bosons=N_ph_max)
    L = nols.liouvillian(dg, a_ph, Hs, g_ph, Vs, Vt, Gamma_s, Gamma_t, kappa, kT)

    # population block numerically
    index_h, index_e = [], []
    for i in range(3):
        index_h.append(
            nols.get_vectorized_rho_index(n1=i, q1=0, n2=i, q2=0, max_bosons=N_ph_max)
        )
        index_e.append(
            nols.get_vectorized_rho_index(n1=i, q1=1, n2=i, q2=1, max_bosons=N_ph_max)
        )
    index = index_h + index_e

    Lpop_num = L[np.ix_(index, index)]

    # population block anallytically
    # Franck-Condon
    n = np.arange(3)
    F = nfc.FC(n, n, g_ph)
    F2 = F**2

    # according to eq S155
    GammaS = Gamma_s * F2
    GammaT = Gamma_t * F2

    GammaP = np.array(
        [
            [GammaS[0, 0], GammaS[0, 1], GammaT[0, 2] + GammaS[0, 2]],
            [0, GammaS[1, 1], GammaS[1, 2]],
            [0, 0, GammaS[2, 2]],
        ]
    )

    GammaM = np.array(
        [
            [GammaT[0, 0], GammaT[0, 1] + GammaS[0, 1], GammaT[0, 2] + GammaS[0, 2]],
            [GammaT[1, 0], GammaT[1, 1], GammaT[1, 2] + GammaS[1, 2]],
            [0, GammaT[2, 1], GammaT[2, 2]],
        ]
    )

    # Eq S155
    Lpop_an = np.array(
        [
            [-GammaP[0, 0], kappa, 0, GammaM[0, 0], GammaM[0, 1], GammaM[0, 2]],
            [0, -kappa, 2 * kappa, GammaM[1, 0], GammaM[1, 1], GammaM[1, 2]],
            [0, 0, -2 * kappa, 0, GammaM[2, 1], GammaM[2, 2]],
            [
                GammaP[0, 0],
                GammaP[0, 1],
                GammaP[0, 2],
                -GammaM[0, 0] - GammaM[1, 0],
                kappa,
                0,
            ],
            [0, GammaP[1, 1], GammaP[1, 2], 0, -kappa, 2 * kappa],
            [0, 0, GammaP[2, 2], 0, 0, -2 * kappa],
        ]
    )

    # numerical vs anallytical  L population block
    assert np.allclose(Lpop_num.real, Lpop_an, atol=(Gamma_t / kappa))

    # checking L coherences blocks

    # S74
    e0_h1 = nols.get_vectorized_rho_index(n1=0, q1=1, n2=1, q2=0, max_bosons=N_ph_max)
    assert np.allclose(L[e0_h1, e0_h1], 1j * hw_ph - (kappa * g_ph**2 / 2 + kappa) / 2)

    # S75
    h0_e1 = nols.get_vectorized_rho_index(n1=0, q1=0, n2=1, q2=1, max_bosons=N_ph_max)
    assert np.allclose(L[h0_e1, h0_e1], 1j * hw_ph - (kappa * g_ph**2 / 2 + kappa) / 2)

    # S76 -77
    h0_e0 = nols.get_vectorized_rho_index(n1=0, q1=0, n2=0, q2=1, max_bosons=N_ph_max)
    h1_e1 = nols.get_vectorized_rho_index(n1=1, q1=0, n2=1, q2=1, max_bosons=N_ph_max)

    assert np.allclose(L[h0_e0, h0_e0], -kappa * g_ph**2 / 4, atol=Gamma_t / kappa)
    assert np.allclose(L[h0_e0, h1_e1], -kappa)  # why - kappa? should be +
    assert np.allclose(
        L[h1_e1, h1_e1], -kappa * g_ph**2 / 4 - kappa, atol=Gamma_t / kappa
    )

    # S78 - 79
    h0_h1 = nols.get_vectorized_rho_index(n1=0, q1=0, n2=1, q2=0, max_bosons=N_ph_max)
    e0_e1 = nols.get_vectorized_rho_index(n1=0, q1=1, n2=1, q2=1, max_bosons=N_ph_max)

    assert np.allclose(L[h0_h1, h0_h1], 1j * hw_ph - kappa / 2)
    assert np.allclose(L[e0_e1, e0_e1], 1j * hw_ph - kappa / 2)

    # virtual transitions

    Gamma_t00_1 = Gamma_t * nfc.FC(0, 0, g_ph) * nfc.FC(1, 1, g_ph)
    Gamma_s00_1 = Gamma_s * nfc.FC(0, 0, g_ph) * nfc.FC(1, 1, g_ph)
    assert np.allclose(L[h0_h1, e0_e1], Gamma_t00_1)
    assert np.allclose(L[e0_e1, h0_h1], Gamma_s00_1)

    # coherences zero?
    rho_st = nme.stationary(L)
    coherences = rho_st - np.diag(np.diag(rho_st))
    max_off = np.max(np.abs(coherences))
    assert max_off < 1e-15

    # cross terms in correlations are zero

    M, _ = nme.regression_theorem(a_ph.d, L, ng, cutoff=0)
    assert np.allclose(M, 0)
    M, _ = nme.regression_theorem(ng, L, a_ph, cutoff=0)
    assert np.allclose(M, 0)
