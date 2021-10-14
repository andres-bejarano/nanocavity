import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import scipy.integrate as integrate

#b=1/ k_BT
b=10

def fermi_dirac(e,mu):
    return 1/(np.exp((e-mu)*b)+1)
 

#(0,e,g,2)=(0,1,2,3)
delta=1.
#here tunneling rates for \Gamma_{Le,lg,re,rg}
#tunneling rate as semicircle with wide w 
def semi_circle(e, mu, w):
    e = np.array(e).reshape(-1, 1)
    mu = np.array(mu).reshape(1, -1)
    x = (e - mu) / w # broadcasting                                                                                                                                                                                
    x = np.clip(x, -1, 1) # values outside the band width should be set to zero
    y = (1 - x * 2) * 0.5
    return np.squeeze(y)

lg=1.
le=1.
rg=1.
re=1.
w=10

#I am taking only 3 points in order to visualize arrays. 
VL = VR = np.linspace(-3, 3, 3)
tu_Lg = lg * semi_circle(0, VL,w)
tu_Le = le * semi_circle(delta, VL,w)
tu_Rg = rg * semi_circle(0, VR,w)
tu_Re = re * semi_circle(delta, VR,w)

print('tu_Lg',tu_Lg)

#transition rates
#\Gamma_{\alpa,j}^{+}
#Gamma_{\alpha,j}^{-}=tun_{\alpha,j} -tun_{\alpha,j}
tun_Lg = tu_Lg*fermi_dirac(0, VL)
tun_Le = tu_Le*fermi_dirac(delta, VL)
tun_Rg = tu_Rg*fermi_dirac(0, VR)
tun_Re = tu_Re*fermi_dirac(delta, VR)

print('tun_Lg',tun_Lg)
#\Gamma_{\alpha,electrons or holes} 
#each element of this matrix host all possible values of
#tun_{\alpha(e, g)}
gamma_le = gamma_re = np.zeros((len(VL),4,4))
gamma_le[:,1,0] = gamma_le[:,3,2] = tun_Lg
gamma_le[:,2,0] = gamma_le[:,3,1] = tun_Le
gamma_re[:,1,0] = gamma_re[:,3,2] = tun_Rg
gamma_re[:,2,0] = gamma_re[:,3,1] = tun_Re

#pending to implements symmetries here in order to save code-lines
gamma_lh = gamma_rh = np.zeros((len(VL),4,4))
gamma_lh[:,0,1] = gamma_lh[:,1,3] = tu_Lg-tun_Lg
gamma_lh[:,0,2] = gamma_lh[:,2,3] = tu_Le-tun_Le
gamma_rh[:,0,1] = gamma_rh[:,1,3] = tu_Rg-tun_Rg
gamma_rh[:,0,2] = gamma_rh[:,2,3] = tu_Re-tun_Re
#print(gamma_le,gamma_lh)


gamma = gamma_lh+gamma_rh+gamma_le+gamma_re
#print('gamma',gamma)

#we need diagonal elements which are the sum of each column
for i in range(0,4):
    gamma[:,i,i] = -gamma.sum(axis=1)[:,i]
#we apply the fact that \sum_i p_i=1
gamma[:,3,:] = 1 
#print('gamma',gamma)

# solve eq 9.22 under above constraint
populations = b = np.zeros((len(VL),4,1))
b[:,3,:]=1
populations = np.linalg.solve(gamma,b)
print('p',populations)
 
# finally we solve 9.28
GL = gamma_le-gamma_lh
print('GL',GL)
I = np.matmul(GL,populations)
I = I.sum(axis=1)
print('current',I)

#visualization
#plt.rc('text', usetex=True)
#plt.rc('font', family='Bitstream Vera Serif', size=16)
#fig = plt.figure(figsize=(8, 6))
#axes = plt.axes()

# 2d visualization


#VL, VR = np.meshgrid(VL, VR)
#plt.contourf(VL ,VR, I,20,cmap='RdGy')
#plt.colorbar(label = ''r'$I/e\Gamma$')
#axes.set_xlabel(r'$eV_L$ ')
#axes.set_ylabel(r'$eV_R$')