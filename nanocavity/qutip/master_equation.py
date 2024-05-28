import numpy as np
from scipy.linalg import eig
import nanocavity.qutip.operators as qo
import nanocavity.master_equation as nme
from qutip import operator_to_vector, steadystate

def eig_norm(L):
    return nme.eig_norm(L.full())

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



def cumulants(CA_p=[], CA_m=[], CL_p=[], CL_m=[], CR_p=[], CR_m=[], p=1e-5):
    #the convergence of this method its highly senstive to x
    #x=np.linspace(-1, 1, 3) does not work
    x = p * np.linspace(-2, 2, 5)
    Emax_xy = np.zeros((len(x), len(x)), dtype=complex)
    L = 0
    for i, xx in enumerate(x):
        for j, yy in enumerate(x):
            if CA_p != []:
                L += qo.liouvillian(CA_p, chi=-xx)
                L += qo.liouvillian(CA_m, chi=xx)
            if CL_p != []:
                L += qo.liouvillian(CL_p, chi=-yy)
                L += qo.liouvillian(CL_m, chi=yy)
            if CR_p != []:
                L += qo.liouvillian(CR_p, chi=-yy)
                L += qo.liouvillian(CR_m, chi=yy)
            Exy, _ = L.eigenstates()
            index = np.argmax(np.real(Exy))
            Emax_xy[i, j] =  Exy[index]
    Ig = np.gradient(np.imag(Emax_xy[:, 2]), x)[2]
    Ie = np.gradient(np.imag(Emax_xy[2, :]), x)[2]
    Zg = -np.gradient(np.gradient(np.real(Emax_xy[:, 2]), x), x)[2]
    Ze = -np.gradient(np.gradient(np.real(Emax_xy[2, :]), x), x)[2]
    return Ig, Ie, Zg, Ze
