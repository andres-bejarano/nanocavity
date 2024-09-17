import numpy as np
from nanocavity.distributions import *


def test_fermi_dirac():
    assert fermi_dirac(0, 0.1) == 0.5
    e = [-1e5, -100, 0, 100, 1e5]
    nf = [1, 1, 0.5, 0, 0]
    assert np.allclose(fermi_dirac(e, 0.1), nf)
    assert np.allclose(fermi_dirac(e, 0.01), nf)
    assert not np.allclose(fermi_dirac(e, 100), nf)


def test_fermi_dirac_arrays():
    e = np.linspace(-100, 100, 10).reshape((2, 5))
    nf = fermi_dirac(e, 0.1)
    assert nf.shape == e.shape
    # 1 - nF(-e) == nF(e)
    nf2 = 1 - fermi_dirac(-e, 0.1)
    assert np.allclose(nf, nf2)
    nf = fermi_dirac(0, 0.1, mu=e)
    assert nf.shape == e.shape
    nf = fermi_dirac(e, 1)
    assert nf.shape == e.shape
    nf = fermi_dirac(0, 1, e)
    assert nf.shape == e.shape
    nf = fermi_dirac(e, 1, e)
    assert nf.shape == e.shape


def test_bose_einstein():
    assert bose_einstein(1e5, 0.1) < 1e-100
    e = [-1e5, -100, 0, 100, 1e5]
    nb = [-1, -1, np.inf, 0, 0]
    assert np.allclose(bose_einstein(e, 0.1), nb)
    assert np.allclose(bose_einstein(e, 0.01), nb)
    assert not np.allclose(bose_einstein(e, 100), nb)


def test_bose_einstein_arrays():
    e = np.linspace(-100, 100, 10).reshape((2, 5))
    nb = bose_einstein(e, 0.1)
    assert nb.shape == e.shape
    # 1 + nB(-e) == -nB(e)
    nb2 = 1 + bose_einstein(-e, 0.1)
    assert np.allclose(-nb, nb2)


def test_bath_dist():
    for bath in ["bosonic", "fermionic", "leadtolead"]:
        distin = bath_dist(E=-1, kT=1, rate="in", bath=bath)(-1)
        distout = bath_dist(E=1, kT=1, rate="out", bath=bath)(1)
        r = distout + distin
        if bath == "bosonic":
            assert np.allclose(r, 1 + 2 * distin)
        elif bath == "fermionic":
            assert np.allclose(r, 1)
        elif bath == "leadtolead":
            distin = bath_dist(E=-1, kT=1, rate="in", bath=bath)(-1)
            distout = bath_dist(E=-1, kT=1, rate="out", bath=bath)(-1)
            assert np.allclose(distin, distout)


def test_lorentz():
    assert lorentzian(-np.inf, 0.1) == 0.0
    assert lorentzian(np.inf, 0.1) == 0.0
    assert np.allclose(lorentzian(0.0, 0.1), 2 / (0.1 * np.pi))


def test_semi_circle():
    e = np.linspace(-2, 2, 3)
    sc = semi_circle(e, 0, 1)
    assert sc[0] == 0
    assert sc[1] == 1
    assert sc[2] == 0
    sc = semi_circle(e, mu=e, w=1)
    assert sc.shape == (3, 3)


def test_Fermi_cb():
    e = np.linspace(-2, 2, 3)
    x = Fermi_cb(e, 0.1)
    assert np.allclose(x, [4.12230725e-09, 0.1, 2.0])
