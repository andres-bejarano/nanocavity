import numpy as np
import nanocavity.distributions as ndist
from secondquant.operator import Operator


def collapses(A_op, H, kT, bath, mu=0, total=True, cutoff=1e-12):
    '''
        Function to calculate the collapse operators which are needed to
        build a Liouvillian with secondquant operators
        Parameters:
            ----
            A_op: secondquantoperator
                annihilation operator 
            H:  secondquant operator 
                Hamiltonian of the central system
            kT: float
                Temperature
            bath: string
                Either 'fermionic' or 'bosonic'
            mu: float
                chemical potential
            total: logical
                Switch wether to return the sum of collapse operators or
                the individual opperators
            cutoff: float
                cutoff for the considered transition matrix elements
    '''
    E, V = H.eigh()

    # Transition matrix elements between final (f) and initial (i) states
    # Thus the first index refers to the final state 
    M_fi = A_op.inner(V)
    dim = A_op.shape[0]
    # Matrix of all energy differences in the system between final and initial
    # state
    E_fi = E.reshape(dim, 1) - E.reshape(1, dim)
    if bath == 'bosonic':
        # This rate is for photon absorption, thus the final state must be 
        # higher in energy than the initial one
        nb_fi_p = np.where(E_fi > 0, ndist.bose_einstein(E_fi, kT), 0)
        # This rate is for photon emission, thus the final state must be 
        # lower in energy than the initial one
        nb_fi_m = np.where(E_fi < 0, 1 + ndist.bose_einstein(-E_fi, kT), 0)
    elif bath == 'fermionic':
        fd_fi_p = ndist.fermi_dirac(E_fi, kT, mu)
        fd_fi_m = 1-fd_fi
    cp, cm = [], []
    for f in range(dim):
        for i in range(dim):
            if abs(M_fi[f, i]) > cutoff:
                P = M_fi[f, i] * \
                        V[:, f].reshape(dim, 1) @ V[:, i].reshape(1, dim)
                if bath == 'bosonic':
                    cp.append(np.sqrt(nb_fi_p[f, i]) * P.conj().T)
                    cm.append(np.sqrt(nb_fi_m[f, i]) * P)

                elif bath == 'fermionic':
                    cp.append(np.sqrt(fd_fi_p[f, i]) * P.conj().T)
                    cm.append(np.sqrt(fd_fi_m[f, i]) * P)
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

