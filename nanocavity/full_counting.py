import numpy as np


def d1(E, zero, p=1e-3):
    L = (E[zero] - E[zero-1]) / p
    R = (E[zero+1] - E[zero]) / p
    M = (E[zero+1] - E[zero-1]) / (2*p)
    return L, R, M

def d2(E, zero, p=1e-3):
    L = (E[zero] + E[zero-2] - 2 * E[zero-1]) / p ** 2
    R = (E[zero+2] + E[zero] - 2 * E[zero+1]) / p ** 2
    M = (E[zero+2] + E[zero-2] - 2 * E[zero]) / (4 * p ** 2)
    return L, R, M

def dxy(E, zero, p=1e-3):
    L = (E[zero-1, zero-1] + E[zero, zero] - E[zero, zero-1] - E[zero-1, zero]) / p ** 2
    R = (E[zero+1, zero+1] + E[zero, zero] - E[zero, zero+1] - E[zero+1, zero]) /  p ** 2
    M = (E[zero+1, zero+1] + E[zero-1, zero-1] - E[zero+1, zero-1] - E[zero-1, zero+1]) / (4 * p ** 2)
    return L, R, M


def M_matrix(Kp, Km, GpL, GmL, GpR, GmR, p=1e-3, ninter=10):
    x = np.linspace(-5*p, 5*p, ninter + 1)
    
    #K[dim(h), dim(h)]
    K = Kp + Km
    K = K[np.newaxis, np.newaxis]
    #K[vl, vr, dim(h), dim(h)]

    #GpL[vl, dim(H), dim(H)]
    #Gpr[vr, dim(H), dim(H)]
    vr, k, _ = GpR.shape
    vl, _, _ = GpL.shape
    GpR = GpR.reshape(1, vr, k, k)
    GmR = GmR.reshape(1, vr, k, k)
    GpL = GpL.reshape(vl, 1, k, k)
    GmL = GmL.reshape(vl, 1, k, k)
    GR = GpR + GmR
    GL = GpL + GmL


    #we write the diagonal
    Gamma = (GL + GR) + K
    Gout = np.zeros(Gamma.shape)
    #Gamma[vl, vr, dim(H), dim(H)]
    for i in range(Gamma.shape[2]):
        Gout[:, :, i, i] = -Gamma.sum(axis=2)[:, :, i]
    
    #we define couting fields with [x=phfield, y=x=elfield, vl, vr, dim(H), dim(H)]
    xp = np.exp(1j * x)[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis] 
    xm = np.exp(-1j * x)[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
    yp = np.exp(1j * x)[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis] 
    ym = np.exp(-1j * x)[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis] 
    Km = Km[np.newaxis, np.newaxis]
    Kp = Kp[np.newaxis, np.newaxis]
    GmR = GmR[np.newaxis, np.newaxis]
    GpR = GpR[np.newaxis, np.newaxis]
    Gout = Gout[np.newaxis, np.newaxis]
    GL = GL[np.newaxis, np.newaxis]
    M = Gout + GL + Km * xp + Kp * xm + GmR * yp + GpR * ym

    #M[x, y, vl, vr, dim(H), dim(H)]
    E, _ = np.linalg.eig(M)
    #E[x, y, vl, vr, dim(H)]
    return E, x

def E_max(Kp, Km, GpL, GmL, GpR, GmR, p=1e-3, ninter=10):
    E, x = M_matrix(Kp, Km, GpL, GmL, GpR, GmR, p, ninter)
    #E[x, y, vl, vr, dim(H)]
    #we look the position of (x,y) = (0,0)
    zero = x.size // 2
    #we look at the max real eigenvalue of dim(H)Xdim(H) matrix 
    #for each vl and vr and x=0, y=0 
    #fixed x and y the axis 2 is dim(H)
    n = np.argmax(np.real(E[zero, zero]), axis=2)
    #we need this index for x,y map
    #n[vl, vr]
    #we repeat len(x),len(y) times in the direction of x,y respectively
    ny = np.repeat(n[np.newaxis], len(x), axis=0)
    nxy = np.repeat(ny[np.newaxis], len(x), axis=0)
    #nx[len(y),vl, vr] and nxy[len(x), len(y), vl, vr]
    E_max = np.take_along_axis(E, np.expand_dims(nxy, axis=4), axis=4)[:, :, :, :, 0]
    return E_max, zero

def cumulants(Kp, Km, GpL, GmL, GpR, GmR, p, ninter=10):
    E, zero = E_max(Kp, Km, GpL, GmL, GpR, GmR, p, ninter)
    #E[x, y, vl, vr]
    E_re = np.real(E)
    E_im = np.imag(E)

    #the first cumulants is the gradient respect to x and y
    #the first cumulants are the average number of particles 
    #per unit T that get in the drain.
    _, _, Nph = d1(E_im[:, zero, :, :], zero, p)
    _, _, Nel = d1(E_im[zero, :, :], zero, p)
    
    #second cumulant Z = sigma2
    _, _, Zph = d2(E_re[:, zero, :, :], zero, p)
    _, _, Zel = d2(E_re[zero, :, :, :], zero, p)
    _, _, covariance = dxy(E_re, zero, p)
    
    return Nph, Nel, -Zph, -Zel, -covariance
