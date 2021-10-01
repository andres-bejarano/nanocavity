import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import scipy.integrate as integrate

#b=1/ k_BT
b=10

def fermi_dirac(e,mu):
    return 1/(np.exp((e-mu)*b)+1)
 

#(0,e,g,2)=(0,1,2,3)
ed=0.
delta=1.
#here tunneling rates for \Gamma_{Le,lg,re,rg}
#tunneling rate as semicircle with wide w 
def semi_circle(e, mu, w):
    e = np.array(e).reshape(-1, 1)
    mu = np.array(mu).reshape(1, -1)
    x = (e - mu) / w # broadcasting                                                                                                                                                                                
    x = np.clip(x, -1, 1) # values outside the band width should be set to zero
    y = (1 - x ** 2) ** 0.5
    return np.squeeze(y)

lg=1.
le=1.
rg=1.
re=1.
w=10

VL = VR = np.linspace(-12, 12, 11)
tu_Lg = lg * semi_circle(ed, VL,w)
tu_Le = le * semi_circle(ed, VL,w)
tu_Rg = rg * semi_circle(ed, VR,w)
tu_Re = re * semi_circle(ed, VR,w)

#transition rates
#\Gamma_{\alpa,j}^{+}
#Gamma_{\alpha,j}^{-}=tun_{\alpha,j} -tun_{\alpha,j}
fL = fermi_dirac(ed, VL).reshape(1, -1)
fR = fermi_dirac(ed, VR).reshape(-1, 1)
tun_Lg = (lg * semi_circle(ed, VL,w)).reshape(1, -1)*\
    (fermi_dirac(ed, VL).reshape(1, -1))
tun_Le = ( le * semi_circle(delta, VL,w))*\
    fermi_dirac(delta, VL).reshape(1, -1)
tun_Rg = ( rg * semi_circle(ed, VR,w))\
    *fermi_dirac(ed, VR).reshape(1, -1)
tun_Re = ( re * semi_circle(delta, VL,w))\
    *fermi_dirac(delta, VR).reshape(1, -1)


#\Gamma_{\alpha,electrons or holes} 
gamma_le = gamma_re = np.zeros((4,4,len(VL),len(VL)))
gamma_le[1,0] = gamma_le[3,2] = tun_Lg
gamma_le[2,0] = gamma_le[3,1] = tun_Le
gamma_re[1,0] = gamma_re[3,2] = tun_Rg
gamma_re[2,0] = gamma_re[3,1] = tun_Re

#pending to implements symmetries here in order to save code-lines
gamma_lh = gamma_rh = np.zeros((4,4,len(VL),len(VL)))
gamma_lh[0,1] = gamma_lh[1,3] = tu_Lg-tun_Lg
gamma_lh[0,2] = gamma_lh[2,3] = tu_Lg-tun_Le
gamma_rh[0,1] = gamma_rh[1,3] = tu_Rg-tun_Rg
gamma_rh[0,2] = gamma_rh[2,3] = tu_Re-tun_Re

#print(gamma_le,gamma_lh)


gamma = gamma_lh+gamma_rh+gamma_le+gamma_re

#we need diagonal elements which are the sum of each column
for i in range(0,3):
    gamma[i,i] = -gamma.sum(axis=0)[0]
#we apply the fact that \sum_i p_i=1
gamma[3,:,:,:] = 1 
#print(gamma)

populations = b = np.zeros((4,1,len(VL),len(VL)))
b[3,:,:,:]=1
#populations = np.linalg.solve(gamma,b)

print('p',populations)
 
I = (gamma_le-gamma_lh)*populations

print(gamma[:,:,1,1])

#print(I(1,0))
#visualization
#plt.rc('text', usetex=True)
#plt.rc('font', family='Bitstream Vera Serif', size=16)
#fig = plt.figure(figsize=(8, 6))
#axes = plt.axes()

# 2d visualization


#VL, VR = np.meshgrid(VL, VR)
#plt.contourf(VL ,VR, I(VL,VR),20,cmap='RdGy')
#plt.colorbar(label = ''r'$I/e\Gamma$')
#axes.set_xlabel(r'$eV_L$ ')
#axes.set_ylabel(r'$eV_R$')

    