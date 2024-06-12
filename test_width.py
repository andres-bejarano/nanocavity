import numpy as np
import nanocavity.rate_equation as nre
from secondquant.composite import composite
import matplotlib.pyplot as plt
##
# General parameter definition
E0 = 0.3
Delta = 0.9
U = 2
kT = 1e-2

gamma_t = 1e-3 * np.eye(2)
gamma_s = 1e-3 * np.eye(2)
gamma0 = gamma_t[0, 0] + gamma_s[0, 0]

kappa = 1e-3
g_ph = 1e-3

V_fin = 1
V = np.linspace(-3, 3, V_fin)
# V = np.array([2, 2.5])
# V = np.array([2.5])
V = 2
mu_t = V
mu_s = -V
omega = np.linspace(1e-5, 2, 100000)
##

[cg, ce, a_ph], [ng, ne, n_ph] = composite(fermion_modes=2, boson_modes=[1])

H = E0 * ng + (E0+Delta) * ne + U*ng*ne \
        + g_ph * (a_ph.d*cg.d*ce + a_ph*ce.d*cg) + n_ph
Een, Est = H.eigsh(H.shape[0])

Gpt, Gmt = nre.transition_rate(Een, Est, [cg, ce], gamma_t, kT=kT, mu=mu_t)
Gps, Gms = nre.transition_rate(Een, Est, [cg, ce], gamma_s, kT=kT, mu=mu_s)
Kp, Km = nre.transition_rate(Een, Est, a_ph, kappa, kT=kT, bath='bosonic')

Gt = Gpt + Gmt
Gs = Gps + Gms
K = Kp + Km

Gamma2 = Gt + Gs + Kp + Km
Gamma = Gt[None, :] + Gs[:, None] + K[None, None]
P = nre.populations(Gamma)
P2 = nre.populations(Gamma2)
I_tip = nre.electro_current2(Gpt-Gmt, P)
# Photon_em = nre.power_spectrum(Kp, Km, P, Een, omega)
r_plus = Kp + Gpt + Gps
r_minus = Km + Gmt + Gms
Photon_em2 = nre.power_spectrum2(Kp, Km, P2, Een, omega)
Gp = Gpt + Gps
Gm = Gmt + Gms
Photon_em3 = nre.power_spectrum3(Kp, Km, Gp, Gm, P2, Een, omega)
Rp = Kp + Gpt + Gps
Rm = Km + Gmt + Gms
Photon_em4 = nre.power_spectrum4(Rp, Rm, Km, Kp, P2, Een, omega)
plot_photon = 0
if plot_photon == 1:
    plt.figure()
    plt.plot(omega, Photon_em3, 'k')
    plt.plot(omega, Photon_em4, 'r--')
    plt.plot(omega, Photon_em2)
    plt.xlim(0.99, 1.01)
    plt.show()
##
# Photon_em_diag = np.zeros((omega.shape[0], V_fin))
# for i in range(V_fin):
#     Photon_em_diag[:, i] = Photon_em[:, i, -1-i]
##

# fig = plt.figure(figsize=(8, 6))
# plt.rc('font', family='Bitstream Vera Serif', size=16)
# plt.rc('text', usetex=True)
# ax = plt.axes()

# Omega, Mu_t = np.meshgrid(omega, mu_t)
# # Im1 = ax.contourf(Omega, Mu_t, Photon_em_diag.T, 100, cmap='viridis')
# fig.colorbar(Im1)
##
plot_photon = 0
if plot_photon == 1:
    plt.figure()
    plt.plot(omega, Photon_em3)
    plt.show()
##
plot_curr = 0
if plot_curr == 1:
    plt.figure()
    plt.plot(V, I_tip/gamma0)
    plt.show()
