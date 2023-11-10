import numpy as np
import nanocavity.distributions as ndist
import nanocavity.operators as no
import nanocavity.rate_equation as nre
from qutip import (Qobj, about, basis, qeye, tensor, steadystate)
import matplotlib.pyplot as plt

Eg = 0.4
delta = 0.9
omega = 1
coupling= 0.3

gamma = 1e-3 * np.eye(2)
g = gamma[0, 0]
kappa =  1
kT = 1e-2
VL = np.linspace(0, 2, 101)
VR = 0

#nanocav-populations
Hnc, La, Ld = no.H_tls_nc(Eg, delta, omega, coupling)
Enc, Vnc = Hnc.eigh()
GpL, GmL = nre.transition_rate(Enc, Vnc, Ld, gamma, mu=VL, kT=kT)
GpR, GmR = nre.transition_rate(Enc, Vnc, Ld, gamma, mu=VR, kT=kT)
GL, GR = nre.transition_rate_matrix(GpL + GmL, GpR + GmR)

#damping matrix
Kp, Km = nre.transition_rate(Enc, Vnc, La, kappa, kT=kT, bath='bosonic')
K = Kp + Km

#transtion rates matrix
Gamma = K[np.newaxis, np.newaxis] + GL + GR
Pnc = nre.populations(Gamma)[:, 0, :]

#QuTiP populations
Hqt, L = no.H_tls_QuTiP(Eg, delta, omega,  coupling)
Eqt, Vqt = Hqt.eigenstates()

def populations_qt(L, Hqt, Eqt, Vqt, VL, VR, kappa, gamma):
    collapses_cavity = no.bosonic_collapses(L[0], E=Eqt, V=Vqt, kT=1e-2, k=kappa)
    collapses_ground = no.fermionic_collapses(L[1], E=Eqt, V=Vqt, VL=VL, VR=VR, kT=1e-2, g=g)
    collapses_excited = no.fermionic_collapses(L[2], E=Eqt, V=Vqt, VL=VL, VR=VR, kT=1e-2 ,g=g)
    c_ops = collapses_cavity + collapses_ground + collapses_excited
    A = steadystate(Hqt, c_ops).full()
    return  np.linalg.eigh(A)[0]

Pqt = np.zeros((len(VL), len(Eqt)))
for i in range(len(VL)):
    Pqt[i, :] = populations_qt(L, Hqt, Eqt, Vqt, VL=VL[i], VR=VR, kappa=kappa, gamma=g)


fig = plt.figure(figsize=(8, 6))
plt.rc('font', family='Bitstream Vera Serif', size=16)
plt.rc('text', usetex=True)
ax = plt.axes()

plt.plot(VL, Pnc)
plt.plot(VL, Pqt)

plt.show()
