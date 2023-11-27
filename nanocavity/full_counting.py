import numpy as np

def transition_rates_fourier(Kp, Km, GpL, GmL, GpR, GmR, p=1e-3):
    x = p * np.linspace(-2, 2, 5)
    
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

def E_max(Kp, Km, GpL, GmL, GpR, GmR, p=1e-3):
    E, x = transition_rates_fourier(Kp, Km, GpL, GmL, GpR, GmR, p)
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
    return E_max, zero, x

def cumulants(Kp, Km, GpL, GmL, GpR, GmR, p):
    E, zero, x = E_max(Kp, Km, GpL, GmL, GpR, GmR, p)
    #E[x, y, vl, vr]
    dx = np.gradient(E, x, axis=0)
    dy = np.gradient(E, x, axis=1)
    dxx = np.gradient(dx[:, zero], x, axis=0)[zero]
    dyy = np.gradient(dy[zero], x, axis=0)[zero]
    dxy = np.gradient(dx[zero], x, axis=0)[zero]

    #the first cumulants is the gradient respect to x and y
    #the first cumulants are the average number of particles 
    #per unit T that get in the drain.
    Iph = np.imag(dx[zero, zero])
    Iel = -np.imag(dy[zero, zero])
    
    #second cumulant Z = sigma2/T
    Zph = -np.real(dxx)
    Zel = -np.real(dyy)
    covariance = -np.real(dxy)
    
    return Iph, Iel, Zph, Zel, covariance
