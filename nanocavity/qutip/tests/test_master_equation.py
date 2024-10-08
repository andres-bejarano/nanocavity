import numpy as np
import nanocavity.qutip.tls as qtls
import nanocavity.tls as tls
import nanocavity.operators as no
import nanocavity.master_equation as nme
import qutip as qt
import pytest


@pytest.fixture(scope="module", params=[0])
def Eg(request):
    return request.param


@pytest.fixture(scope="module", params=[0.99])
def Delta(request):
    return request.param


@pytest.fixture(scope="module", params=[2])
def U(request):
    return request.param


@pytest.fixture(scope="module", params=[1e-3, 1e-2])
def g_ph(request):
    return request.param


@pytest.fixture(scope="module", params=[1])
def hw_ph(request):
    return request.param


@pytest.fixture(scope="module", params=[3])
def max_photons(request):
    return request.param


@pytest.fixture(scope="module", params=[False])
def rwa(request):
    return request.param


@pytest.fixture(scope="module", params=[1e-1])
def kT(request):
    return request.param


@pytest.fixture(scope="module", params=[1e-3, 1e-1])
def kappa(request):
    return request.param


@pytest.fixture(scope="module", params=[1e-3])
def Gamma(request):
    return request.param


@pytest.fixture(scope="module", params=[3])
def VL(request):
    return request.param


@pytest.fixture(scope="module", params=[-2])
def VR(request):
    return request.param


@pytest.fixture(scope="module", params=[[0]])
def tlist(request):
    return request.param


def test_g2(
    Eg, Delta, U, g_ph, hw_ph, max_photons, rwa, kT, Gamma, kappa, VL, VR, tlist
):
    # Calculating g2 with nanocavity
    H0_nc, Hint_nc, anni_ops_nc = tls.Hamiltonian(
        Eg, Delta, hw_ph, g_ph, U, max_photons, rwa
    )
    a_ph_nc = anni_ops_nc[2]
    c_ops_nc = tls.collapses(H0_nc, anni_ops_nc, VL, VR, kappa, Gamma, Gamma, kT)
    _, c_am = no.collapses(a_ph_nc, H0_nc, kT, "bosonic", kappa, total=False)
    L_nc = no.liouvillian(H0_nc + Hint_nc, c_ops_nc)
    Jam = no.jump(c_am)
    g2_nc = nme.g2(L_nc, Jam, [0], verbose=False)

    H0, Hint, anni_ops = qtls.Hamiltonian(Eg, Delta, hw_ph, g_ph, U, max_photons, rwa)
    a_ph = anni_ops[2]
    c_ops = qtls.collapses(H0, anni_ops, VL, VR, kappa, Gamma, Gamma, kT)
    for method in ["direct", "eigen", "svd", "power"]:
        rho = qt.steadystate(H0 + Hint, c_ops, method="direct")
        g2, _ = qt.coherence_function_g2(
            H0 + Hint, rho, tlist, c_ops, a_ph, options={"progress_bar": False}
        )
    assert np.allclose(g2_nc, g2)
