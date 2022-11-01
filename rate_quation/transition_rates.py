import numpy as np
from secondquant.composite import *
from secondquant.operator import Operator
import numpy.linalg as la
import matplotlib.pyplot as plt

#fermi distribution
def fermi(E, T=10e-1, mu=0):
    return 1. / (np.exp((E - mu)/T) + 1.)

#chemical potential
V = np.linspace(-3, 3, 100)
N = len(V)

#leads-exciton couplings
GL = np.array([1, 0.4, 0.4, 1])
GR = np.array([1, 0.4, 0.4, 1])

#hamiltonian parameters
g = 0.2
delta = 1.0
hwc = 1.0

#hamiltonian diagonzalization
d, Nf, a, Nb = composite(fermion_modes=2, boson_modes=1, max_bosons=50)
eigen = 6

H0 = delta*Nf[1] + hwc*Nb[0]
Ht = g * (a[0].d+a[0]) * (d[0].d*d[1] + d[1].d*d[0])
H = H0 + Ht
E, v = H.eigsh(k=eigen, which='SA')
v = np.array(v)
E = np.array(E)


# E,v eigenvalues and eigenstates
# d operator which interacts with the  bath
# G tunneling rate between the bath and system
def transition_rate(E, v, d, G, mu):
    h = len(E)
    G = G.reshape(2, 2)
    DE = E.reshape(1, -1, 1) - E.reshape(1, 1, -1)
    mu = mu.reshape(-1, 1, 1)
    M = np.empty((len(d), h, h))
    
    for i in range(len(d)):
        M[i]  = d[i].inner(v)

    Mplus = np.einsum('iab,ij,jba->ab', M.conj(), G, M)
    Gamma_plus = fermi(DE, mu) * \
            np.repeat(Mplus.reshape(1, h, h), mu.shape[2], axis=0) 
    Gamma_minus = (1-fermi(-DE, mu) * \
            np.repeat(Mplus.conj().T.reshape(1, h, h), mu.shape[2], axis=0))
    return Gamma_plus, Gamma_minus

#transition rates matrix
gp, gm = transition_rate(E, v, d, GL, V)

gp = gp.reshape(1, N, len(E), len(E))
gm = gm.reshape(N, -1, len(E), len(E))
gamma = gp + gm

for i in range(len(E)):
    gamma[:, :, i, i] = 0
    gamma[:, :, i, i] = -gamma.sum(axis=2).reshape(len(E) * N**2,1)[i]

#population stationary solution
populations = np.zeros((len(V), len(V), len(E), 1))
b = np.zeros((len(V), len(V), len(E), 1))

gamma[:, : ,len(E) - 1, :] = 1
b[:, :, len(E) - 1, :] = 1

populations =  la.solve(gamma, b)


print(populations.sum())
print(populations.shape)

#current

I = np.empty((len(V),len(V)))

#avoid this#
for i in range(len(V)):
    for j in range(len(V)):
        I[i,j] = (gp - gm)[i,j].dot(populations[i,j]).sum()

#visualization
plt.rc('text', usetex=True)
plt.rc('font', family='Bitstream Vera Serif', size=16)
fig = plt.figure(figsize=(8, 6))
axes = plt.axes()

# 2d visualization
VL, VR = np.meshgrid(V, V)
plt.contourf(VL ,VR, -I, 20, cmap='RdGy')
plt.colorbar(label =''r'$I/e\Gamma$')
axes.set_xlabel(r'$eV_L$')
axes.set_ylabel(r'$eV_R$')

plt.savefig('test.pdf')
