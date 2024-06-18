import numpy as np
import nanocavity.distributions as ndist
from secondquant.operator import Operator

def collapses(A_op, H, kT, bath, mu=0, total=True, cutoff=1e-12):
    if isinstance(A_op, Operator):
        A_op = A_op.toarray()
    
    E, V = H.eigh()
    dim = A_op.shape[0]
    Vinv = np.linalg.inv(V)
    cp, cm = [], []
    for i, Ei in enumerate(E):
        for j, Ej in enumerate(E):
            Mij = V[:, i].T @ A_op @ V[:, j]
            if abs(Mij) > cutoff:
                Eji = Ej - Ei
                P = Mij * V[:,  i].reshape(dim, 1) @ V[:, j].reshape(1, dim)
                if bath=='bosonic':
                    nb = ndist.bose_einstein(Eji, kT=kT)
                    cp.append(np.sqrt(nb) * P.conj().T)
                    cm.append(np.sqrt(1 + nb) * P)
                elif bath=='fermionic':
                    fd = ndist.fermi_dirac(Eji, kT=kT, mu=mu)
                    cp.append(np.sqrt(fd) * P.conj().T)
                    cm.append(np.sqrt(1 - fd) * P)
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
