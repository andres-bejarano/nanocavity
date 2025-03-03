import numpy as np
import nanocavity.qutip.tls as qtls
import nanocavity.tls as tls
import nanocavity.operators as no
import nanocavity.master_equation as nme
import qutip as qt


def test_g2():
    Eg, Delta, U = -0, 0.99, 2
    g_ph = 1e-2
    hw_ph = 1
    max_photons = 1
    rwa = False

    kT = 1e-1
    Gamma = 1e-3
    kappa = 1e-2
    VL = 3
    VR = -VL

    tlist = [0]
    # Calculating g2 with nanocavity
    H0_nc, Hint_nc, anni_ops_nc = tls.Hamiltonian(
        Eg, Delta, hw_ph, g_ph, U, max_photons, rwa
    )
    a_ph_nc = anni_ops_nc[2]
    c_ops_nc = tls.collapses(H0_nc, anni_ops_nc, VL, VR, kappa, Gamma, Gamma, kT, hw_ph)
    basis = H0_nc.eigh()
    _, c_am = no.collapses(a_ph_nc, basis, kT, "bosonic", kappa, total=False)
    L_nc = no.liouvillian(H0_nc + Hint_nc, c_ops_nc)
    rho_st = nme.stationary(L_nc)

    g2_nc = nme.g2(L_nc, a_ph_nc, tlist, verbose=False, rho_st=rho_st)
    g2_zero = nme.g2_zero(a_ph_nc, rho_st)
    assert np.allclose(g2_nc, g2_zero)

    H0, Hint, anni_ops = qtls.Hamiltonian(Eg, Delta, hw_ph, g_ph, U, max_photons, rwa)
    a_ph = anni_ops[2]
    c_ops = qtls.collapses(H0, anni_ops, VL, VR, kappa, Gamma, Gamma, kT, hw_ph)
    rho = qt.steadystate(H0 + Hint, c_ops)
    g2, _ = qt.coherence_function_g2(
        H0 + Hint, rho, tlist, c_ops, a_ph, options={"progress_bar": False}
    )
    assert np.allclose(g2_nc, g2)
