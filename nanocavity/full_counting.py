import numpy as np
import numpy.linalg as la
from secondquant.operator import Operator
from secondquant.composite import *
from nanocavity.distributions import *
from nanocavity.rate_equation import *

def E_matrix(Kp, Km, GL, GR, p):
    x = np.linspace(-2*p, 2*p, 5)
    #K[vl, vr, dim(h), dim(h)]
    K = Kp + Km
    K = K[np.newaxis, np.newaxis]
    Gamma = (GL + GR) + K
    Gout = np.zeros(Gamma.shape)

    #the first two index are V_l and V_r
    #third and fourth are the dimension of hilbert space
    for i in range(Gamma.shape[2]):
        Gout[:, :, i, i] = -Gamma.sum(axis=2)[:, :, i]

    M = (Gout + GL + GR + Kp)[np.newaxis] + Km[np.newaxis] * \
        np.exp(1j * x)[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
    #M[chi, vl, vr, dim(H), dim(H)]
    E, _ = np.linalg.eig(M)
    #E[chi, vl, vr, dim(H)]
    return E, x

def current_fcs(Kp, Km, GL, GR, p):
    E, x = E_matrix(Kp, Km, GL, GR, p)
    #E[chi, vl, vr, dim(H)]
    index = x.size // 2
    #we look at the max real value for chi=0
    index_M = np.argmax(np.real(E[index]), axis=2)
    #we need this index for all chi values
    index_Mr = np.repeat(index_M[np.newaxis], len(x), axis=0)
    E_imag = np.take_along_axis(np.imag(E), np.expand_dims(index_Mr, axis=3), axis=3)[: , :, :, 0]
    #E[chi, vl, vr]
    return (E_imag[index + 1] - E_imag[index]) / p


def variance_fcs(Kp, Km, GL, GR, p):
    E, x = E_matrix(Kp, Km, GL, GR, p)
    index = x.size // 2
    index_M = np.argmax(np.real(E[index]), axis=2)
    index_Mr = np.repeat(index_M[np.newaxis], len(x), axis=0)
    E_real = np.take_along_axis(np.real(E), np.expand_dims(index_Mr, axis=3), axis=3)[: , :, :, 0]
    return -(E_real[index+2] + E_real[index] - 2 * E_real[index+1]) / p ** 2
