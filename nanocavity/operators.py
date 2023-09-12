import numpy as np
from nanocavity.distributions import *
from secondquant.composite import *
from qutip import (qeye, tensor, destroy)

#two level system coupled to single cavity mode
def H_tls_nc(Eg, delta, omega, coupling):
    [d1, d2, a], [Nf1, Nf2, Nb] = \
            composite(fermion_modes=2, boson_modes=1, max_bosons=1)
    H0 = Eg * Nf1 + (Eg +  delta) * Nf2 + omega * Nb
    Hint = coupling * (a.d * d1.d * d2 + a * d2.d * d1)
    H = H0 +  Hint
    La = [a]
    Ld = [d1, d2]
    return H, La, Ld

#two level system coupled to single cavity mode in QuTiP

def H_tls_QuTiP(Eg, delta, omega, coupling):
    N = 2
    dg = tensor(destroy(2), qeye(2), qeye(N))
    de = tensor(qeye(2), destroy(2), qeye(N))
    a = tensor(qeye(2), qeye(2), destroy(N))
    
    H0 = Eg * dg.dag() * dg + (Eg + delta)* de.dag() * de + omega * a.dag() * a
    Hint = coupling * (a.dag() * dg.dag() * de + a * de.dag() * dg)
    H = H0 + Hint
    E, V = H.eigenstates()
    L = [a, dg, de]
    return H, L


#collapses operators

def fermionic_collapses(A_op, E, V, VL, VR, kT, g):
    c = []
    for i in range(len(E)):
        for j in range(len(E)):
            Mij = (V[i].dag() * A_op * V[j]).full().squeeze() ** 2
            if Mij != 0:
                dE = abs(E[i]-E[j])
                fL = fermi_dirac(dE, mu=VL, kT=kT)
                fR = fermi_dirac(dE, mu=VR, kT=kT)
                P = V[i] * V[j].dag()
                c.append(np.sqrt(g * Mij * (1 - fL)) * P)
                c.append(np.sqrt(g * Mij * (1 - fR)) * P)
                c.append(np.sqrt(g * Mij * fL) * P.dag())
                c.append(np.sqrt(g * Mij * fR) * P.dag())
    return c


def bosonic_collapses(A_op, E, V, kT, k):
    c = []
    for i in range(len(E)):
        for j in range(len(E)):
            Mij = (V[i].dag() * A_op * V[j]).full().squeeze() ** 2
            if Mij != 0:
                dE = abs(E[i]-E[j])
                nb = bose_einstein(dE, kT=kT)
                P = V[i] * V[j].dag()
                c.append(np.sqrt(k * Mij * (1 + nb)) * P)
                c.append(np.sqrt(k * Mij * nb) * P.dag())
    return c

