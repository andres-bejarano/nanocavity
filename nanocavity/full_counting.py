import numpy as np
import numpy.linalg as la
from secondquant.operator import Operator
from secondquant.composite import *
from nanocavity.distributions import *
from nanocavity.rate_equation import *

def M_matrix(Kp, Km, GL, GpR, GmR, p):
    x = np.linspace(-2*p, 2*p, 5)
    
    #K[dim(h), dim(h)]
    K = Kp + Km
    K = K[np.newaxis, np.newaxis]
    #K[vl, vr, dim(h), dim(h)]

    #GL[vl, vr, dim(H), dim(H)]
    #Gpr[vr, dim(H), dim(H)]
    vr, k, _ = GpR.shape
    GpR = GpR.reshape(1, vr, k, k)
    GmR = GmR.reshape(1, vr, k, k)
    GR = GpR + GmR
    #GR[vl, vr, dim(H), dim(H)]

    #we write the diagonal
    Gamma = (GL + GR) + K
    Gout = np.zeros(Gamma.shape)
    #Gamma[vl, vr, dim(H), dim(H)]
    for i in range(Gamma.shape[2]):
        Gout[:, :, i, i] = -Gamma.sum(axis=2)[:, :, i]
    
    #we define couting fields with [x=phfield, y=x=elfield, vl, vr, dim(H), dim(H)]
    ph_field = np.exp(1j * x)[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis] 
    el_field = np.exp(1j * x)[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis] 

    M = (Gout + GL + GpR + Kp)[np.newaxis, np.newaxis] +\
            Km[np.newaxis, np.newaxis] * ph_field + GmR[np.newaxis, np.newaxis] * el_field
    #M[x, y, vl, vr, dim(H), dim(H)]
    E, _ = np.linalg.eig(M)
    #E[x, y, vl, vr, dim(H)]
    return E, x

def cumulants(Kp, Km, GL, GpR, GmR, p):
    E, x = M_matrix(Kp, Km, GL, GpR, GmR, p)
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

    #the first cumulants is the gradient respect to x and y
    E_imag = np.take_along_axis(np.imag(E), np.expand_dims(nxy, axis=4), axis=4)[:, :, :, :, 0]
    #E[x, y, vl, vr]
    I_ph = (E_imag[zero+1,  zero] - E_imag[zero, zero]) / p
    I_el = -(E_imag[zero, zero+1] - E_imag[zero, zero]) / p
    
    #second cumulant
    E_real = np.take_along_axis(np.real(E), np.expand_dims(nxy, axis=4), axis=4)[:, :, :, :, 0]
    sigma2_ph = -(E_real[zero+2, zero] + E_real[zero, zero] - 2 * E_real[zero+1, zero]) / p ** 2
    sigma2_el = -(E_real[zero, zero+2] + E_real[zero, zero] - 2 * E_real[zero, zero+1]) / p ** 2
    covarianze = -(E_real[zero+1, zero+1] + E_real[zero, zero] - E_real[zero, zero+1] - E_real[zero+1, zero]) / p ** 2
    
    return I_ph, I_el, sigma2_ph, sigma2_el, covarianze


