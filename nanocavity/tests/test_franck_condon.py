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
        assert np.allclose(FC_0n(0, g), FC_mat[0, 0])
        assert np.allclose(FC_0n(0, g), FC_vec[0])
        assert np.allclose(FC_0n(1, g), FC_mat[0, 1])
        assert np.allclose(FC_0n(1, g), FC_vec[1])
        assert np.allclose(FC_0n(1, g), nfc.FC(0, 1, g))
        assert np.allclose(FC_11(g), FC_mat[1, 1])
        assert np.allclose(FC_21(g), FC_mat[2, 1])
        assert np.allclose(FC_22(g), FC_mat[2, 2])
        assert np.allclose(FC_mat, FC_mat.T)
        # assert np.allclose(FC_vec, FC_vec2)


def test_franck_condon_square():
    g_vec = [0.1, 0.5, 1, 1.5, 2]
    for g in g_vec:
        FC_mat = nfc.FC2(np.arange(6), np.arange(6), g)
        FC_vec = nfc.FC2(np.arange(6), 0, g)
        FC_vec2 = nfc.FC2(0, np.arange(6), g)
        assert np.allclose(FC_0n(0, g) ** 2, FC_mat[0, 0])
        assert np.allclose(FC_0n(0, g) ** 2, FC_vec[0])
        assert np.allclose(FC_0n(1, g) ** 2, FC_mat[0, 1])
        assert np.allclose(FC_0n(1, g) ** 2, FC_vec[1])
        assert np.allclose(FC_0n(1, g) ** 2, nfc.FC2(0, 1, g))
        assert np.allclose(FC_11(g) ** 2, FC_mat[1, 1])
        assert np.allclose(FC_21(g) ** 2, FC_mat[2, 1])
        assert np.allclose(FC_22(g) ** 2, FC_mat[2, 2])
        assert np.allclose(FC_mat, FC_mat.T)
        assert np.allclose(FC_vec, FC_vec2)
