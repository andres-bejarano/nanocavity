import pytest
import nanocavity.franck_condon as nfc
import numpy as np


def FC_0n(n, g):
    return np.exp(-(g**2) / 2) * g**n / np.sqrt(np.math.factorial(n))


def FC_11(g):
    return (1 - g**2) * np.exp(-(g**2) / 2)


def FC_21(g):
    return np.sqrt(2) * g * (1 - g**2 / 2) * np.exp(-(g**2) / 2)


def FC_22(g):
    return (1 - g**2 / 2 + g**4 / 2) * np.exp(-(g**2) / 2)


def test_franck_condon():
    g_vec = [0.1, 0.5, 1, 1.5, 2]
    for g in g_vec:
        assert np.allclose(FC_0n(0, g), nfc.FC_factor(0, 0, g), atol=1e-2)
        assert np.allclose(FC_0n(1, g), nfc.FC_factor(0, 1, g))
        assert np.allclose(FC_11(g), nfc.FC_factor(1, 1, g))
        assert np.allclose(FC_21(g), nfc.FC_factor(2, 1, g))
        assert np.allclose(FC_22(g), nfc.FC_factor(2, 2, g))
    pass
