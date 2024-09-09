import numpy as np
import nanocavity.distributions as ndist
from secondquant.operator import Operator


def collapses(A_op, H, kT, bath, mu=0, total=True, cutoff=1e-12):
    E, V = H.eigh()

    M_ij = A_op.inner(V)
    dim = A_op.shape[0]
    E_ij = E.reshape(dim, 1) - E.reshape(1, dim)
    # E, V = H.eigh()
    if bath == 'bosonic':
        nbij = ndist.bose_einstein(E_ij, kT)
        nbij = np.where(nbij > 0, nbij, 0)
        np.fill_diagonal(nbij, 0)
        print(nbij)
    elif bath == 'fermionic':
        fdij = ndist.fermi_dirac(E_ij, kT)
        fdij = np.where(fdij > 0, fdij, 0)
    cp, cm = [], []
    for i in range(dim):
        for j in range(dim):
            if abs(M_ij[i, j]) > cutoff:
                P = M_ij[i, j] * \
                        V[:, i].reshape(dim, 1) @ V[:, j].reshape(1, dim)
                if bath == 'bosonic':
                    cp.append(np.sqrt(nbij[i, j]) * P.conj().T)
                    cm.append(np.sqrt(1 + nbij[i, j]) * P)

                elif bath == 'fermionic':
                    cp.append(np.sqrt(fdij[i, j]) * P.conj().T)
                    cm.append(np.sqrt(1 - fdij[i, j]) * P)
    if total:
        return cp + cm
    return cp, cm

def jump(c_ops):
    J = 0
    for c in c_ops:
        J += np.kron(c, c.conj())
    return J 

def dissipator(c_ops, method='kron'):
    Id = np.eye(c_ops[0].shape[0])
    #Look https://arxiv.org/pdf/1504.05266
    L = 0
    for c in c_ops:
        cdc = c.conj().T @ c
        if method=='einsum':
            L = np.einsum('ik,jl->ijkl', c, c.conj())
            L -= 0.5 * np.einsum('ik,jl->ijkl', cdc, Id)
            L -= 0.5 * np.einsum('ki,lj->ijkl', Id, cdc)
        elif method=='kron':
            L += np.kron(c, c.conj())
            L -= 0.5 * np.kron(Id, c.conj().T @ c)
            L -= 0.5 * np.kron(c.T @ c.conj(), Id)
    return L

def liouvillian(H, c_ops, method='kron'):
    if isinstance(H, Operator):
        H = H.toarray()
    dim = H.shape[0]
    Id = np.eye(H.shape[0])
    #Writing the coherent evolution
    if method=='einsum':
        L = -1j * (np.einsum('ik,jl->ijkl', H, Id) -  1j * np.einsum('ki,lj->ijkl', Id, H))
        return np.reshape(L, (dim ** 2, dim ** 2))
    elif method=='kron':
        L = 1j * (np.kron(Id, H) - np.kron(H, Id))

    L += dissipator(c_ops, method)    
    return L

