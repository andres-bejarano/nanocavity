import pytest
import numpy as np
import secondquant as sq
import nanocavity.operators as no
import nanocavity.distributions as nd

@pytest.fixture(scope='module', params=[-0.5, 0.5])
def Eg(request):
    return request.param


@pytest.fixture(scope='module', params=[1e-2])
def kT(request):
    return request.param


@pytest.fixture(scope='module', params=[1e-3])
def Gamma(request):
    return request.param


@pytest.fixture(scope='module', params=[-0.5, 0, 0.5])
def V(request):
    return request.param


def test_fermionic_collapse(Eg, kT, Gamma, V):
    dg, ng = sq.composite(1)
    H = Eg*dg.d*dg
    cp, cm = no.collapses(dg, H, kT, 'fermionic', Gamma, V, False)
    cp_an = np.zeros((2, 2))
    cm_an = np.zeros((2, 2))
    cp_an[1, 0] = np.sqrt(Gamma * nd.fermi_dirac(Eg, kT, V))
    cm_an[0, 1] = np.sqrt(Gamma * (1-nd.fermi_dirac(Eg, kT, V)))
    assert np.allclose(cp, cp_an)
    assert np.allclose(cm, cm_an)
