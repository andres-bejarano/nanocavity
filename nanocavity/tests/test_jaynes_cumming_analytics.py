import numpy as np
import nanocavity.jaynes_cumming_analytics as jc

def test_Omega_R():
    
    # Test when coupling is zero, Omega_R should be |hw_ph - Delta|
    assert jc.Omega_R(1.0, 0.5, 0.0) == 0.5
    assert jc.Omega_R(2.0, 1.0, 0.0) == 1.0

    # Test when hw_ph equals Delta, Omega_R should be 2 * sqrt(n) * g_ph
    assert jc.Omega_R(1.0, 1.0, 0.5) == 1.0
    assert jc.Omega_R(2.0, 2.0, 1.0, max_bosons=2) == 2.0 * np.sqrt(2)

def test_Epm():
    
    # Test for rabi splitting
    Em, Ep = jc.Emp((1, 2, 3, 4), 5) 
    OmegaR = jc.Omega_R(2, 3, 4, 5)
    assert np.allclose((Ep - Em), OmegaR)
