# nanocavity

**nanocavity** is a Python package for modeling current-induced light emission in biased plasmonic nanocavities. It solves the Lindblad quantum master equation (within the secular approximation) for open quantum systems coupled to fermionic and bosonic baths, and computes experimentally relevant observables: emission spectra, photon currents, and second-order photon correlation functions.

---

## Physics

The master equation describes the time evolution of the reduced density matrix $\rho$ of an open quantum system coupled to external baths. It is implemented through the Liouvillian superoperator $\mathcal{L}$, which encodes both the coherent (Hamiltonian) dynamics and the dissipative processes induced by the environment:


$$\dot{\rho} = \mathcal{L}\rho = -i[H_S, \rho] + \sum_\omega \Gamma(\omega)\left[A(\omega)\rho A^\dagger(\omega) - \tfrac{1}{2}\{A^\dagger(\omega)A(\omega), \rho\}\right]$$

Here $\omega$ labels the transition energies of the system, $\Gamma(\omega)$ is the transition rate associated with each energy, and $A(\omega)$ are the collapse operators — the quantum jump
operators that describe the dissipative transitions at energy $\omega$.

---

## Features

- Energy-resolved Lindblad collapse operators for fermionic and bosonic baths
- Full Liouvillian superoperator in Liouville space
- Steady-state density matrix (via null-space or linear solve)
- Emission spectrum via spectral decomposition of the Liouvillian (no numerical Fourier transform needed)
- Second-order photon correlation function  via the quantum regression theorem
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
- (optionally `QuTiP >= 5.0`)


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


This package was developed as part of the following work, where the theoretical
framework and numerical implementation are described in detail:

- **Thesis:** Andrés Bejarano, *Quantum theory of photon emission in current-driven single-molecule tunnel junctions*, University of the Basque Country / University of Bordeaux (2025). [PDF](https://addi.ehu.es/bitstream/handle/10810/77573/TESIS_ANDR%C3%89S_FELIPE_BEJARANO_S%C3%81NCHEZ..pdf?sequence=1&isAllowed=y)

- **Paper:** Andrés Bejarano *et al.*, arXiv:2504.13657 (2025). [arXiv](https://arxiv.org/abs/2504.13657)

## Citation

If you use **nanocavity** in your research, please cite this repository:
Andrés Bejarano, nanocavity: a Python package for modeling light emission
in biased plasmonic nanocavities (2025). https://github.com/andres-bejarano/nanocavity
