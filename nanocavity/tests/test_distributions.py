import numpy as np
from nanocavity.distributions import *


def test_fermi(): 
    assert fermi(-100, 0.1) == 1.0
    assert fermi(0, 0.1) == 0.5
    assert fermi(100, 0.1) == 0.0

def test_fermi_arrays():
    e = np.linspace(-1000, 1000, 10).reshape((2, 5))
    nf = fermi(e, 1)
    assert nf.shape == e.shape
    nf = fermi(0, 1, e)
    assert nf.shape == e.shape
    nf = fermi(e, 1, e)
    assert nf.shape == e.shape
