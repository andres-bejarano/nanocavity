import numpy as np
from scipy.linalg import eig
import nanocavity.operators as no
from qutip import operator_to_vector, steadystate

def eig_norm(L):
    El, vl, vr = eig(L.full(), left=True)
    norm = np.einsum("ai,ai->i", vl.conj(), vr) ** -0.5
    vl *= norm
    vr *= norm
    return El, vl, vr

def current(J, L):
    J = J.full()
    #w/v left/right eigenvectors
    El, vl, vr = eig_norm(L)
    index = np.argmin(np.abs(El))
    return np.dot(vl[:, index], np.dot(J, vr[:, index]))

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
