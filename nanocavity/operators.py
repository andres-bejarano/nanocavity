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
            if Mij > cutoff:
                Eji = Ej - Ei
                P = Mij * V[:,  i].reshape(dim, 1) @ V[:, j].reshape(1, dim)
                Pv = Vinv @ P @ V
                if bath=='bosonic':
                    nb = ndist.bose_einstein(Eji, kT=kT)
                    cp.append(np.sqrt(nb) * Pv.conj().T)
                    cm.append(np.sqrt(1 + nb) * Pv)
                elif bath=='fermionic':
                    fd = ndist.fermi_dirac(Eji, kT=kT, mu=mu)
                    cp.append(np.sqrt(fd) * Pv.conj().T)
                    cm.append(np.sqrt(1 - fd) * Pv)
    if total:
        return cp + cm
    return cp, cm

def collapses_tls(H_parameters, VL, VR, kappa, gL, gR, kT, m=0, alone=True, iva=False, Hint=0):

    H, [dg, de, a] = H_tls(*H_parameters)

    if iva:
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
    else:
        S_op =  [dg, de, a]
        return S_op, H, c_ops

def dissipator(c_ops, method='kron'):
    Id = np.eye(c_ops[0].shape[0])
    #Look https://arxiv.org/pdf/1504.05266
    
    for c in c_ops:
        cdc = c.conj().T @ c
        if method=='einsum':
            L = np.einsum('ik,jl->ijkl', c, c.conj())
            L -= 0.5 * np.einsum('ik,jl->ijkl', cdc, Id)
            L -= 0.5 * np.einsum('ki,lj->ijkl', Id, cdc)

        elif method=='kron':
            L = np.kron(c, c.conj())
            L -= 0.5 * np.kron(cdc, Id)
            L -= 0.5 * np.kron(Id, cdc)
    return L

def liouvillian(H, c_ops, method='kron'):
    Id = np.eye(H.shape[0])
    #Writing the coherent evolution
    if method=='einsum':
        L = 1j * np.einsum('ik,jl->ijkl', H, Id)
        L -= 1j * np.einsum('ki,lj->ijkl', Id, H)
        return np.reshape(L, (dim ** 2, dim ** 2))
    elif method=='kron':
        L = 1j * np.kron(H, Id)
        L -= 1j * np.kron(Id, H)

    L += dissipator(c_ops, method)
    
    return L
