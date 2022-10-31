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
g = 0.2
delta = 1.0
hwc = 1.0

#hamiltonian diagonzalization
d, Nf, a, Nb = composite(fermion_modes=2, boson_modes=1, max_bosons=50)
#d, Nf = fermion_fock_space(2)
eigen = 6

H0 = delta*Nf[1] + hwc*Nb[0]
V = g * (a[0].d+a[0]) * (d[0].d*d[1] + d[1].d*d[0])
H = H0 + V
#H = delta * Nf[1]
E, v = H.eigsh(k=eigen, which='SA')
v = np.array(v)
E = np.array(E)
print(len(d))


# E,v eigenvalues and eigenstates
# d operator which interacts with the  bath
# G tunneling rate between the bath and system
def transition_rate(E, v, d, G, mu):
    h = len(E)
    G = np.array(G)
    DE = E.reshape(-1, 1) - E.reshape(1, -1) 
    M = np.empty((len(d), h, h))
    for i in range(len(d)):
        M[i]  = d[i].inner(v)

    Mplus = (M.conj().transpose(0, 1, 2)).dot(M).transpose((0, 2, 1, 3))
    Mplus = Mplus.transpose(2, 3, 0, 1).reshape(h, h, len(d)*2) * G
    Mplus = Mplus.transpose(2, 1, 0).reshape(len(d)*2, h, h)
    Gamma_plus = fermi(DE) * sum(Mplus)
    Gamma_minus = (1-fermi(-DE)) * sum(Mplus).conj().T
    return Gamma_plus, Gamma_minus


gp, gm = transition_rate(E, v, d, GL, V)

gamma = gp + gm

for i in range(len(E)):
    gamma[i, i] = 0
    gamma[i, i] = -gamma.sum(axis=0)[i]



populations = np.zeros((len(E), 1))
b = np.zeros((len(E), 1))

print(gamma.shape)
gamma[len(E)-1, :] = 1
b[len(E)-1, :] = 1

populations =  la.solve(gamma, b)


print(populations.sum())


    

