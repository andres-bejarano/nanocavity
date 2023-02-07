import numpy as np
from nanocavity.distributions import *


def test_fermi(): 
    assert fermi(-100, 0.1) == 1.0
    assert fermi(0, 0.1) == 0.5
    assert fermi(100, 0.1) == 0.0

def test_fermi_arrays():
    e = np.linspace(-100, 100, 10).reshape((2, 5))
    nf = fermi(e, 1)
    assert nf.shape == e.shape
    nf = fermi(0, 1, e)
    assert nf.shape == e.shape
    nf = fermi(e, 1, e)
    assert nf.shape == e.shape

def test_bose():
    assert bose(0, 0.1) == np.inf
    assert bose(100, 0.1) == 0.0

def test_semi_circle():
    e = np.linspace(-2, 2, 3)
    sc = semi_circle(e, 0, 1)
    assert sc[0] == 0
    assert sc[1] == 1
    assert sc[2] == 0
    sc = semi_circle(e, mu=e, w=1)
    assert sc.shape == (3, 3)
