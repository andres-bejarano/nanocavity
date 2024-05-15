import numpy as np
from secondquant.composite import composite2
import nanocavity.rate_equation as nre
import matplotlib.pyplot as plt
##
# General parameter definition
E0 = 0.2
Delta = 0.9
U = 2
kT = 1e-2

gamma_t = 1e-3 * np.eye(2)
gamma_s = 1e-3 * np.eye(2)
gamma0 = gamma_t[0, 0] + gamma_s[0, 0]

kappa = 1
g_ph = 0.3
omega = np.linspace(0, 3, 101)

V_fin = 500
V = np.linspace(-3, 3, V_fin)

mu_t = V
mu_s = V

[cg, ce, a_ph], [ng, ne, n_ph] = composite2(fermion_modes=2, boson_modes=[1])

H = E0 * ng + (E0+Delta) * ne + U*ng*ne \
        + g_ph * (a_ph.d*cg.d*ce + a_ph*ce.d*cg) + n_ph
Een, Est = H.eigsh(H.shape[0])

Gpt, Gmt = nre.transition_rate(Een, Est, [cg, ce], gamma_t, kT=kT, mu=mu_t)
Gps, Gms = nre.transition_rate(Een, Est, [cg, ce], gamma_s, kT=kT, mu=mu_s)
Kp, Km = nre.transition_rate(Een, Est, a_ph, kappa, kT=kT, bath='bosonic')

Gt = (Gpt + Gmt)[None, :]
Gs = (Gps + Gms)[:, None]
K = (Kp + Km)[None, None, :]

Gamma = K + Gt + Gs
P = nre.populations(Gamma)

Ig = nre.power_spectrum(Kp, Km, P, Een, omega)
