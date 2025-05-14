import math

import numpy as np
import pytest

import nanocavity.franck_condon as nfc


# The following functions are taken from: Mitra, Aleiner and Millis, PRB 69, 245302 (2004)
# Title: Phonon effects in molecular transistors: Quantal and classical treatment
def FC_0n(n, g):
    return np.exp(-(g**2) / 2) * g**n / np.sqrt(math.factorial(n))


def FC_11(g):
    return (1 - g**2) * np.exp(-(g**2) / 2)


def FC_21(g):
    return np.sqrt(2) * g * (1 - g**2 / 2) * np.exp(-(g**2) / 2)


def FC_22(g):
    return (1 - g**2 * 2 + g**4 / 2) * np.exp(-(g**2) / 2)


def test_franck_condon():
    g_vec = [0.1, 0.5, 1, 1.5, 2]
    for g in g_vec:
        FC_mat = nfc.FC(np.arange(6), np.arange(6), g)
        FC_vec = nfc.FC(np.arange(6), 0, g)
        FC_vec2 = nfc.FC(0, np.arange(6), g)
        assert np.isclose(FC_0n(0, g), FC_mat[0, 0])
        assert np.isclose(FC_0n(0, g), FC_vec[0])
        assert np.isclose(FC_0n(1, g) ** 2, FC_mat[1, 0] ** 2)
        assert np.isclose(FC_0n(1, g), FC_mat[1, 0])
        assert np.isclose(FC_0n(1, g), -FC_mat[0, 1])
        assert np.isclose(FC_0n(1, g), -FC_vec[1])
        assert np.isclose(FC_0n(1, g), -nfc.FC(1, 0, g))
        assert np.isclose(FC_11(g), FC_mat[1, 1])
        assert np.isclose(FC_21(g), FC_mat[2, 1])
        assert np.isclose(FC_22(g), FC_mat[2, 2])
        assert not np.allclose(FC_mat, FC_mat.T)
        assert not np.allclose(FC_vec, FC_vec2)
        assert np.allclose(FC_mat**2, FC_mat.T**2)
        assert np.allclose(FC_vec**2, FC_vec2**2)


def test_compare_koch_yar():
    g_vec = [0.1, 0.5, 1, 1.5, 2]
    for g in g_vec:
        FC_koch = nfc.FC(np.arange(6), np.arange(6), g)
        FC_yar = nfc.FC(np.arange(6), np.arange(6), g, method="Yar")
        assert np.allclose(FC_koch, FC_yar)


def test_sign_FC(method):
    g = 0.01  # weak coupling limit, expand the matrixelement
    for i in range(5):
        if method == None:
            assert nfc.FC(i, i + 1, g) > 0
            assert nfc.FC(i + 1, i, g) < 0
        else:
            assert nfc.FC(i, i + 1, g, method=method) > 0
            assert nfc.FC(i + 1, i, g, method=method) < 0


@pytest.fixture(scope="module", params=[None, "Koch", "Yar"])
def method(request):
    return request.param


def test_negative_input(method):
    g = 1
    with pytest.raises(Exception):
        if method == None:
            nfc.FC(1, -1, g)
            nfc.FC(-1, 1, g)
            nfc.FC(-1, -1, g)
        else:
            nfc.FC(1, -1, g, method=method)
            nfc.FC(-1, 1, g, method=method)
            nfc.FC(-1, -1, g, method=method)


def test_array_input(method):
    g = 1
    n = np.arange(5)
    if method == None:
        assert len(nfc.FC(n, n, g).shape) == 2
        assert len(nfc.FC(1, n, g).shape) == 1
        with pytest.raises(Exception):
            nfc.FC(1, 1, g).shape
    else:
        assert len(nfc.FC(n, n, g, method=method).shape) == 2
        assert len(nfc.FC(1, n, g, method=method).shape) == 1
        with pytest.raises(Exception):
            nfc.FC(1, 1, g, method=method).shape


def test_float_input(method):
    g = 0.01
    with pytest.raises(Exception):
        if method == None:
            nfc.FC(1.3, 1, g)
            nfc.FC(1, 1.3, g)
            nfc.FC(1.3, 1.3, g)
        else:
            nfc.FC(1.3, 1, g, method=method)
            nfc.FC(1, 1.3, g, method=method)
            nfc.FC(1.3, 1.3, g, method=method)


def test_g_imag():
    g = 0.3 + 0.4j
    with pytest.raises(Exception):
        nfc.FC(1, 1, g, method="Koch")
        nfc.FC(1, 1, g)
    nfc.FC(1, 1, g, method="Yar")


def test_fc_method():
    with pytest.raises(Exception):
        nfc.FC(1, 1, 0.5, method="no valid method")


def test_probabilities():
    n = np.arange(25)
    m = np.arange(3)
    M = nfc.FC(m, n, 1.5)
    s = np.sum(M**2, axis=0)
    assert np.allclose(s, 1)


def test_transpose():
    n = np.arange(10)
    X = nfc.FC(n, n, -0.01)
    Y = nfc.FC(n, n, 0.01)
    assert np.allclose(X, Y.T)
