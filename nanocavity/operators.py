import numpy as np
import nanocavity.distributions as ndist
import secondquant as sq
from secondquant.operator import Operator

#two level system coupled to single cavity mode
def H_tls(Eg, delta, omega, coupling, u=0, rwa=True, max_bosons=1, ret_nop=False):
    [dg, de, a], [Nfg, Nfe, Nb] = \
        sq.composite(fermion_modes=2, boson_modes=1, max_bosons=max_bosons)
    He = Eg * Nfg + (Eg +  delta) * Nfe + u * dg.d * de.d * de * dg
    Hp = omega * Nb
    H0 = He + Hp
    if rwa:
        Hint = coupling * (a.d * dg.d * de + a * de.d * dg)
    else:
        Hint = coupling * (a + a.d) * (dg.d * de + de.d * dg)
    H = H0 +  Hint
    L = [dg, de, a]
    if ret_nop:
        return H, L, [Nfg, Nfe, Nb]
    return H, L

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

def collapses_tls(H_parameters, VL, VR, kappa, gL, gR, kT, alone=True, iva=False):

    H, [dg, de, a] = H_tls(*H_parameters)

    if iva:
        coupling = H_parameters[3]
        Hint = coupling * (a.d * dg.d * de + a * de.d * dg)
        H -= Hint
    #left electrode
    c_gL = collapses(dg, H, kT, bath='fermionic', mu=VL)
    c_eL = collapses(de, H, kT, bath='fermionic', mu=VL)
    CL = np.sqrt(gL) * np.array(c_gL + c_eL)

    #right electrode
    c_gR = collapses(dg, H, kT, bath='fermionic', mu=VR)
    c_eR = collapses(de, H, kT, bath='fermionic', mu=VR)
    CR = np.sqrt(gR) * np.array(c_gR + c_eR)

    #cavity mode
    CA = collapses(a, H, kT, bath='bosonic')

    CA = np.sqrt(kappa) * np.array(CA)

    c_ops = np.concatenate((CL, CR, CA))

    if alone:
        return c_ops
    if iva:
        return [dg, de, a], H + Hint, c_ops
    return [dg, de, a], H, c_ops

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
    Id = np.eye(H.shape[0])
    #Writing the coherent evolution
    if method=='einsum':
        L = -1j * (np.einsum('ik,jl->ijkl', H, Id) -  1j * np.einsum('ki,lj->ijkl', Id, H))
        return np.reshape(L, (dim ** 2, dim ** 2))
    elif method=='kron':
        L = 1j * (np.kron(Id, H) - np.kron(H, Id))

    L += dissipator(c_ops, method)
    
    return L
