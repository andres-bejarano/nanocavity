import numpy as np
import pytest
import qutip as qt

import nanocavity.operators as no
import nanocavity.qutip.tls as nqtls
import nanocavity.tls as ntls


def test_liouvillian_basic():
    Eg, Delta, hw_ph, g_ph, U = 0.1, 0.9, 1.1, 0.05, 1.2
    max_bosons = 3

    Hnc0, Hnc1, _ = ntls.Hamiltonian(Eg, Delta, hw_ph, g_ph, U, max_bosons)
    Hqt0, Hqt1, _ = nqtls.Hamiltonian(Eg, Delta, hw_ph, g_ph, U, max_bosons)

    d = Hnc0.shape[0]
    d4 = (d, d, d, d)

    # check coherent evolution
    for c in [0, 1]:
        L_nc = no.liouvillian(Hnc0 + c * Hnc1).reshape(d4)
        # note that qutip uses F-order memory layout
        L_qt = qt.liouvillian(Hqt0 + c * Hqt1, []).full()
        assert np.allclose(L_nc, L_qt.reshape(d4, order="F"))
        assert not np.allclose(L_nc, L_qt.reshape(d4, order="C"))
