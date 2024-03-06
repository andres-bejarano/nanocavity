import numpy as np
from scipy.linalg import eig
import nanocavity.operators as no


def current(J, L):
    J = J.full()
    #w/v left/right eigenvectors
    El, vl, vr = eig(L.full(), left=True)
    

    norm = np.einsum("ai,ai->i", vl.conj(), vr) ** -0.5
    vl *= norm
    vr *= norm

    index = np.argmin(np.abs(El))
    return np.dot(vl[:, index], np.dot(J, vr[:, index]))


def cumulants(H, S_op, VL, VR,  kT=1e-2, kappa=0.1, gL=1e-3, gR=1e-3, m=0, iva=False, p=1e-5):
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

def noise(L, Jin, Jout, wlist, method='eigen'):
    J1 = Jout.full() - Jin.full()
    J2 =  Jout.full() + Jin.full()

    #w/v left/right eigenvectors
    El, vl, vr = eig(L.full(), left=True)


    norm = np.einsum("ai,ai->i", vl.conj(), vr) ** -0.5
    vl *= norm
    vr *= norm

    index = np.argmin(np.abs(El))
    J2tr =  np.dot(vl[:, index], np.dot(J2, vr[:, index]))
    
    S = np.zeros(len(wlist))
    for j, w in enumerate(wlist):
        Sj = 0
        for i, Ei in enumerate(El[1:]):
            num =  Ei * (np.dot(vl[:, 0].T.conj(), np.dot(J1, vr[:, i]))) \
                * (np.dot(vl[:, i].T.conj(), np.dot(J1, vr[:, 0])))
            den = w ** 2 + np.abs(Ei) ** 2    
            Sj += np.real(num)/den
        S[j] = Sj
    return J2tr + 2 * S
