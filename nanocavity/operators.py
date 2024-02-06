import numpy as np
import nanocavity.distributions as ndist
import secondquant.composite as sc
from qutip import (qeye, tensor, destroy)

#two level system coupled to single cavity mode
def H_tls_nc(Eg, delta, omega, coupling, rwa=True, max_bosons=1, ret_nop=False):
    [dg, de, a], [Nfg, Nfe, Nb] = \
            sc.composite(fermion_modes=2, boson_modes=1, max_bosons=max_bosons)
    H0 = Eg * Nfg + (Eg +  delta) * Nfe + omega * Nb
    if rwa:
        Hint = coupling * (a.d * dg.d * de + a * de.d * dg)
    else:
        Hint = coupling * (a + a.d) * (dg.d * de + de.d * dg)
    H = H0 +  Hint
    L = [dg, de, a]
    if ret_nop:
        return H, L, [Nfg, Nfe, Nb]
    return H, L

#two level system coupled to single cavity mode in QuTiP

def H_tls_QuTiP(Eg, delta, omega, coupling, rwa=True, max_bosons=1):
    N = max_bosons + 1
    dg = tensor(destroy(2), qeye(2), qeye(N))
    de = tensor(qeye(2), destroy(2), qeye(N))
    a = tensor(qeye(2), qeye(2), destroy(N))
    
    H0 = Eg * dg.dag() * dg + (Eg + delta)* de.dag() * de + omega * a.dag() * a
    if rwa:
        Hint = coupling * (a.dag() * dg.dag() * de + a * de.dag() * dg)
    else:
        Hint = coupling * (a + a.dag()) * (dg.dag() * de + de.dag() * dg)
    H = H0 + Hint
    E, V = H.eigenstates()
    L = [dg, de, a]
    return H, L


#collapses operators

def fermionic_collapses(A_op, E, V, VL, VR, kT, gL, gR):
    c = []
    for i, Ei in enumerate(E):
        for j, Ej in enumerate(E):
            Mij = A_op.matrix_element(V[i], V[j]) ** 2
            if Mij != 0:
                Eij = Ei-Ej
                fLm = 1 - ndist.fermi_dirac(Eij, mu=VL, kT=kT)
                fRm = 1 - ndist.fermi_dirac(Eij, mu=VR, kT=kT)
                fLp = ndist.fermi_dirac(-Eij, mu=VL, kT=kT)
                fRp = ndist.fermi_dirac(-Eij, mu=VR, kT=kT)
                P = (V[i] * V[j].dag()).transform(V)
                c.append(np.sqrt(gL * Mij * fLm) * P)
                c.append(np.sqrt(gR * Mij * fRm) * P)
                c.append(np.sqrt(gL * Mij * fLp) * P.dag())
                c.append(np.sqrt(gR * Mij * fRp) * P.dag())
    return c


def bosonic_collapses(A_op, E, V, kT, k):
    c = []
    for i, Ei in enumerate(E):
        for j, Ej in enumerate(E):
            Mij = A_op.matrix_element(V[i], V[j]) ** 2 
            if Mij != 0:
                Eij = Ei-Ej
                nb = 1 + ndist.bose_einstein(Eij, kT=kT)
                nb = ndist.bose_einstein(-Eij, kT=kT)
                P = (V[i] * V[j].dag()).transform(V)
                c.append(np.sqrt(k * Mij * (1 + nb)) * P)
                c.append(np.sqrt(k * Mij * nb) * P.dag())
    return c

def lead_cavity_lead_collapses(A_op, E, V, VL, VR, kT, m):
    c = []
    for i, Ei in enumerate(E):
        for j, Ej in enumerate(E):
            Mij = A_op.matrix_element(V[i], V[j]) ** 2
            if Mij != 0:
                dE = Ei-Ej
                dist1 = ndist.Fermi_cb(VL-VR-dE, kT)
                dist2 = ndist.Fermi_cb(VR-VL-dE, kT)
                dist = dist1 + dist2
                coef = np.sqrt(m * dist * Mij) 
                P = (V[i] * V[j].dag()).transform(V)
                c.append(coef * P)
    return c
