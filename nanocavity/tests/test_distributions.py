import numpy as np
from nanocavity.distributions import *


def test_fermi_dirac(): 
    assert fermi_dirac(-100) == 1.0
    assert fermi_dirac(0) == 0.5
    assert fermi_dirac(100) == 0.0
    assert fermi_dirac(-100, 0.1) == 1.0
    assert fermi_dirac(0, 0.1) == 0.5
    assert fermi_dirac(100, 0.1) == 0.0

def test_fermi_dirac_arrays():
    e = np.linspace(-100, 100, 10).reshape((2, 5))
    nf = fermi_dirac(e)
    assert nf.shape == e.shape
    nf = fermi_dirac(0, mu=e)
    assert nf.shape == e.shape
    nf = fermi_dirac(e, 1)
    assert nf.shape == e.shape
    nf = fermi_dirac(0, 1, e)
    assert nf.shape == e.shape
    nf = fermi_dirac(e, 1, e)
    assert nf.shape == e.shape

def test_bose_einstein():
    assert bose_einstein(0) == np.inf
    assert bose_einstein(100) == 0.0
    assert bose_einstein(0, 0.1) == np.inf
    assert bose_einstein(100, 0.1) == 0.0

def test_semi_circle():
    e = np.linspace(-2, 2, 3)
    sc = semi_circle(e, 0, 1)
    assert sc[0] == 0
    assert sc[1] == 1
    assert sc[2] == 0
    sc = semi_circle(e, mu=e, w=1)
    assert sc.shape == (3, 3)
