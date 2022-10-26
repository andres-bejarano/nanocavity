import numpy as np
from secondquant.composite import *
from secondquant.operator import Operator
import numpy.linalg as la

#fermi distribution
def fermi(E, T=10e-3, mu=0):
    return 1. / (np.exp((E -mu)/T) + 1.)

#leads-exciton couplings
GL = np.array([1, 0.4, 0.4, 1])
GR = np.array([1, 0.4, 0.4, 1])

#hamiltonian paramertes
g = 0
delta = 1.0
hwc = 1.0

#hamiltonian diagonzalization
d, Nf, a, Nb = composite(fermion_modes=2, boson_modes=1, max_bosons=50)
eigen = 7

H0 = delta*Nf[1] + hwc*Nb[0]
V = g * (a[0].d+a[0]) * (d[0].d*d[1] + d[1].d*d[0])
H = H0 + V
E, v = H.eigsh(k=eigen, which='SA')
v = np.array(v)
E = np.array(E)

# E,v eigenvalues and eigenstates
# d operator which interacts with the  bath
# G tunneling rate between the bath and system
def transition_rate(E, v, d, G):
    G = G.reshape(len(d), len(d))
    DE = E.reshape(-1, 1) - E.reshape(1, -1)
    Mplus = 0
    Mminus = 0
    #pending to avoid this loop
    for i in range(len(d)):
        for j in range(len(d)):
            M1 = v.transpose().dot((d[i].d).dot(v))
            M2 =  v.transpose().dot((d[j]).dot(v))
            Mplus += G[i,j] * M1 * M2.transpose()
            Mminus += G[i,j] * M2 * M1.transpose()
    Gamma_plus = fermi(DE) * Mplus
    Gamma_minus = (1-fermi(-DE)) * Mminus
    return Gamma_plus, Gamma_minus


gamma_p, gamma_m = transition_rate(E, v, d, GL)

gamma = gamma_p + gamma_m


for i in range(len(gamma)):
    gamma[i,i] = 0
    gamma[i,i] = -gamma.sum(axis=0)[i]

populations = np.zeros((len(gamma), 1))
b = np.zeros((len(gamma), 1))

gamma[len(gamma)-1, :] = 1
b[len(gamma)-1, :] = 1

populations =  la.solve(gamma,b)




print(populations.sum())
    

