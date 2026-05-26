# nanocavity

**nanocavity** is a Python package for modeling current-induced light emission in biased plasmonic nanocavities. It solves the Lindblad quantum master equation (within the secular approximation) for open quantum systems coupled to fermionic and bosonic baths, and computes experimentally relevant observables: emission spectra, photon currents, and second-order photon correlation functions g^(2)(tau).

---

## Physics

The package targets driven-dissipative systems where electronic transport through a molecular junction excites a plasmonic cavity mode. The dynamics are governed by the Lindblad master equation in Lindblad form, with the secular approximation applied when constructing the dissipator. This ensures complete positivity of the density matrix.

---

## Features

- Energy-resolved Lindblad collapse operators for fermionic and bosonic baths
- Full Liouvillian superoperator in Liouville space
- Steady-state density matrix (via null-space or linear solve)
- Emission spectrum via spectral decomposition of the Liouvillian (no numerical Fourier transform needed)
- Second-order photon correlation function g^(2)(tau) via the quantum regression theorem
- Electronic and photonic current from jump superoperators

---

## Installation

```bash
source install.sh
```

## Dependencies

- `numpy >= 1.13`
- `scipy >= 1.5.0`
- [`secondquant >= 0.4.2`]
- `matplotlib`

---

## Quick example

```python
import nanocavity.operators as no
import nanocavity.master_equation as nme

# Define Hamiltonian and collapse operators
H0, Hint, a = no.Hamiltonian(params, sys)
c_ops_e, c_ops_ph = no.collapses(H0, params.env)

# Build Liouvillian and steady state
L = no.liouvillian(H0 + Hint, c_ops_e + c_ops_ph)
rho = nme.stationary(L)

# Emission spectrum and photon correlations
S  = nme.spectrum(L, a, rho)
g2 = nme.g2(L, a, rho)
```
