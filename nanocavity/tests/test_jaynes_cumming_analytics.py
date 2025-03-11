import numpy as np

import nanocavity.jaynes_cumming_analytics as jc


def test_Omega_R():

    # Test when coupling is zero, Omega_R should be |hw_ph - Delta|
    assert jc.Omega_R(1.0, 0.5, 0.0) == 0.5
    assert jc.Omega_R(2.0, 1.0, 0.0) == 1.0

    # Test when hw_ph equals Delta, Omega_R should be 2 * sqrt(n) * g_ph
    assert jc.Omega_R(1.0, 1.0, 0.5) == 1.0
    assert jc.Omega_R(2.0, 2.0, 1.0, max_bosons=2) == 2.0 * np.sqrt(2)


def test_Epm():

    # Test for rabi splitting
    Em, Ep = jc.Emp((1, 2, 3, 4), 5)
    OmegaR = jc.Omega_R(2, 3, 4, 5)
    assert np.allclose((Ep - Em), OmegaR)


def test_theta():

    # Test when detuning goes to zero

    theta1 = jc.theta(1, 1, 3)
    theta2 = jc.theta(1, 3, 0)
    assert np.allclose(theta1, np.pi / 4)
    assert np.allclose(theta2, 0)


def test_check_degeneracy():
    A = np.array([2, 3, 4, 5, 2])
    B = np.array([2, 3, 1, 5, 6])

    assert jc.check_degeneracy(A) == True
    assert jc.check_degeneracy(B) == False


def test_E_index():

    # Test for bare
    H_parameters_bare = (1.0, 2.0, 1.5, 0.1, 3.2)
    Elist_bare = np.array([0, 1.5, 1, 2.5, 3, 4.5, 7.2, 8.7])
    expected_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    result = jc.E_index(H_parameters_bare, Elist_bare, states="bare")
    assert np.allclose(result, expected_indices)
