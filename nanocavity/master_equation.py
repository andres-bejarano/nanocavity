import numpy as np
from scipy.linalg import eig
import nanocavity.operators as no
from secondquant.operator import Operator
from qutip import operator_to_vector, steadystate

def eig_norm(L):
    El, vl, vr = eig(L, left=True)
    norm = np.einsum("ai,ai->i", vl.conj(), vr) ** -0.5
    vl *= norm
    vr *= norm
    return El, vl, vr

def eigen_operator(A, v):
    if isinstance(A, Operator):
        A = A.toarray()
    #First we save the matrix elements of A operator
    C = np.einsum('ki,ij,pj->kp', v.conj().T, A, v)
    #Now we calculate the operator asociated to each matrix element
    #The specification of index kp determines a single eigenoperator |l><p|
    Op = np.einsum('ki,pj->ijkp',  v.conj().T,  v)
    return Op * C

def lindblad(A_op, method='einsum'):
    Id = np.eye(A_op.shape[0])
    #Look https://arxiv.org/pdf/1504.05266

    if method=='einsum':
        L = np.einsum('ik,jl->ijkl', A_op, A_op.conj())
        L -= 0.5 * np.einsum('ia,ak,jl->ijkl', A_op.conj().T, A_op, Id)
        L -= 0.5 * np.einsum('ki,la,aj->ijkl', Id, A_op.conj().T, A_op.conj())
        return L
    
    elif method=='kron':
        L = np.kron(A_op, A_op.conj())
        L -= 0.5 * np.kron(A_op.conj().T @ A_op, Id)
        L -= 0.5 * np.kron(Id, A_op.conj().T @ A_op.conj())
        return L

def liouvillian(H, System_op, gt, gs, kappa, method='einsum', iva=False, Hint=0):

    [dg, de, a] = System_op
    dim = H.shape[0]
    Id = np.eye(dim)
   
    if iva:
        H -= Hint
    E, V = H.eigh()
    
    Vinv = np.linalg.inv(V)


    #Dissipators, we need the jumps wich are the eigenoperators
    Dg = eigen_operator(dg, V)
    De = eigen_operator(de, V) 
    A = eigen_operator(a, V)

    #for each eigenoperator we calculate one dissipator
    L = 0j
    #dissipator in the large bias limit
    for i in range(dim):
        for j in range(dim):
            Dgij = Vinv @ Dg[:, :, i, j] @ V
            Deij = Vinv @ De[:, :, i, j] @ V
            Aij = Vinv @ A[:, :, i, j] @ V
            
            #Adding electrons from a given lead
            L += gt * lindblad(Dgij.conj().T, method) 
            L += gt * lindblad(Deij.conj().T, method)
            
            #Removing electrons from a given lead
            L += gs * lindblad(Dgij, method) 
            L += gs * lindblad(Deij, method)
            
            #cavity dissipation
            L += kappa * lindblad(Aij, method) 

    if iva:
        H += Hint
        Hd = np.einsum('ijkp->ij', eigen_operator(H, V)) 
    
    else:
        Hd = E * np.eye(dim)

    #Writing the coherent evolution
    if method=='einsum':
        L -= 1j * np.einsum('ik,jl->ijkl', Hd, Id)
        L += 1j * np.einsum('ki,lj->ijkl', Id, Hd)
        return np.reshape(L, (dim ** 2, dim ** 2))
    elif method=='kron':
        L -= 1j * np.kron(Hd, Id) 
        L += 1j * np.kron(Id, Hd)
        return L

def stationary(L):
    #E, V = eig(L)
    E, V = np.linalg.eig(L)
    # find the zero-eigenvalue mode index
    idx0 = np.argmin(np.abs(E))
    return V[:, idx0] / V[:, idx0].reshape(H.ndim, H.ndim).trace()


def current(J, L):
    J = J.full()
    #w/v left/right eigenvectors
    El, vl, vr = eig_norm(L)
    index = np.argmin(np.abs(El))
    return np.dot(vl[:, index], np.dot(J, vr[:, index]))


def cumulants(H, S_op, VL, VR,  kT=1e-2, kappa=0.1, gL=1e-3, gR=1e-3, m=0, iva=False, p=1e-5):
    #the convergence of this method its highly senstive to x
    #x=np.linspace(-1, 1, 3) does not work
    x = p * np.linspace(-2, 2, 5)
    Emax_xy = np.zeros((len(x), len(x)), dtype=complex)
    for i, xx in enumerate(x):
        for j, yy in enumerate(x):
            Lxy = no.Liouvillian(H, S_op, VL, VR, kT, kappa, gL, gR, m, iva, chi_b=xx, chi_f=yy)
            Exy, _ = Lxy.eigenstates()
            index =  np.argmax(np.real(Exy))
            Emax_xy[i, j] =  Exy[index]
    Ig = np.gradient(np.imag(Emax_xy[:, 2]), x)[2]
    Ie = np.gradient(np.imag(Emax_xy[2, :]), x)[2]
    Zg = -np.gradient(np.gradient(np.real(Emax_xy[:, 2]), x), x)[2]
    Ze = -np.gradient(np.gradient(np.real(Emax_xy[2, :]), x), x)[2]
    return Ig, Ie, Zg, Ze

def noise(L, Jin, Jout, wlist=[0], method='direct'):
    J1 = Jout.full() - Jin.full()
    J2 = Jout.full() + Jin.full()

    if method=='direct':
        rho_ss = steadystate(L)
        d = rho_ss.shape[0]
        rho_ss = operator_to_vector(rho_ss).full()
        L = L.full()
        d2 = L.shape[0]
        Id = np.eye(d2, d2)
        #we have to solve tr{ALB^-1C} 
        #A = J' = J - Tr{J * rho_st}
        Jtr1 = np.dot(J1, rho_ss).reshape(d, d).trace()
        if Jtr1 == 0:
            return [0]
        else:
            A = J1 - Jtr1 * Id
            #C = J' rho_st
            C = np.dot(A, rho_ss)

            #we can avoid inverting B: B^-1 C = W 
            #BW = C  we can compute W as:
            #B = L^2 + \omega^2
            S = np.zeros(len(wlist))
            for i, omega in enumerate(wlist):
                B = np.dot(L, L) + Id * omega ** 2
                W = np.linalg.solve(B, C)
                S[i] = np.real(np.dot(A, np.dot(L, W)).reshape(d, d).trace())
            Jtr2 = np.dot(J2, rho_ss).reshape(d, d).trace()
            return  Jtr2 - 2 * S

    elif method=='eigen':
        print('The stability of this method it is not guaranteed')
        #w/v left/right eigenvectors
        El, vl, vr = eig_norm(L)
        index = np.argmin(np.abs(El))
        J2tr =  np.dot(vl[:, index], np.dot(J2, vr[:, index]))
        S = np.zeros(len(wlist))
        for j, w in enumerate(wlist):
            Sj = 0
            for i, Ei in enumerate(El[1:]):
                num =  Ei * (np.dot(vl[:, 0].T.conj(), np.dot(J1, vr[:, i]))) \
                          * (np.dot(vl[:, i].T.conj(), np.dot(J1, vr[:, 0])))
                den = w ** 2 + np.abs(Ei) ** 2    
                Sj += num/den
            S[j] = np.real(Sj)
        return J2tr - 2 * S
