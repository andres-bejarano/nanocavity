import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import scipy.integrate as integrate

#b=1/ k_BT
b=10

def fermi_dirac(e,mu):
    return 1/(np.exp((e-mu)*b)+1)
 

#(0,e,g,2)=(0,1,2,3)
#e_g = 0 
delta = 0.

#here tunneling rates for \Gamma_{Le,lg,re,rg}
#tunneling rate as semicircle with wide w 
def semi_circle(e, mu, w):
    e = np.array(e).reshape(-1, 1)
    mu = np.array(mu).reshape(1, -1)
    x = (e - mu) / w # broadcasting                                                                                                                                                                                
    x = np.clip(x, -1, 1) # values outside the band width should be set to zero
    y = (1 - x ** 2) ** 0.5
    return np.squeeze(y)

lg,le,rg,re=1,1,1,1

w=10e100

VL = VR = np.linspace(-12,12, 101)
N = len(VL)
tu_Lg = lg * semi_circle(0, VL,w)
tu_Le = le * semi_circle(delta, VL,w)
tu_Rg = rg * semi_circle(0, VR,w)
tu_Re = re * semi_circle(delta, VR,w)


#transition rates
#\Gamma_{\alpa,j}^{+}
#Gamma_{\alpha,j}^{-}=tun_{\alpha,j} -tun_{\alpha,j}
tun_Lg = tu_Lg*fermi_dirac(0, VL)
tun_Le = tu_Le*fermi_dirac(delta, VL)
tun_Rg = tu_Rg*fermi_dirac(0, VR)
tun_Re = tu_Re*fermi_dirac(delta, VR)


#\Gamma_{\alpha,electrons or holes} 
#each element of this matrix host all possible values of
#tun_{\alpha(e, g)}
gamma_le = gamma_re = np.zeros((4,4,N))
gamma_le[1,0] = gamma_le[3,2] = tun_Lg
gamma_le[2,0] = gamma_le[3,1] = tun_Le
gamma_re[1,0] = gamma_re[3,2] = tun_Rg
gamma_re[2,0] = gamma_re[3,1] = tun_Re

#pending to implements symmetries here in order to save code-lines
gamma_lh = gamma_rh = np.zeros((4,4,N))
gamma_lh[0,1] = gamma_lh[1,3] = tu_Lg-tun_Lg
gamma_lh[0,2] = gamma_lh[2,3] = tu_Le-tun_Le
gamma_rh[0,1] = gamma_rh[1,3] = tu_Rg-tun_Rg
gamma_rh[0,2] = gamma_rh[2,3] = tu_Re-tun_Re

#total gamma is the sum of electrodes and holes gamma
gamma_l = (gamma_le+gamma_lh)
gamma_r = (gamma_re+gamma_rh)


#we need diagonal elements which are the sum of each column
#imposing eq 9.26
for i in range(0,4):
    gamma_l[i,i] = -gamma_l.sum(axis=0)[i]
    gamma_r[i,i] = -gamma_r.sum(axis=0)[i]
    
#since we have two variables we need to built a 2d map
#hence, each matrix element of total gamma will be a 
#len(VL)x\len(VR) 
gamma_l = gamma_l.reshape(4,4,N,1)
gamma_r = gamma_r.reshape(4,4,1,N)

gamma = gamma_l+gamma_r



#we apply the fact that \sum_i p_i=1
gamma[3,:,:,:] = 1 
# solve eq 9.22 under above constraint
populations = b = np.zeros((4,1,N,N))
b[3,:]=1

#Pending to avoid this loop, problems with linalg.solve manual
for i in range(N):
    for j in range(N):
        populations[:,:,i,j] = \
            la.solve(gamma[:,:,i,j],b[:,:,i,j])

#If I want to check that sum_i P_i = 1
#print(populations.sum(axis=0))


# finally we solve 9.28
GL = gamma_le-gamma_lh
GR = gamma_re-gamma_rh
I_L = I_R = np.zeros((N,N))

#Pending to avoid this loop
for i in range(N):
    for j in range(N):
        I_L[i,j] = np.matmul(GL[:,:,i],populations[:,:,i,j]).sum()
        I_R[i,j] = np.matmul(GR[:,:,i],populations[:,:,i,j]).sum()


#visualization
plt.rc('text', usetex=True)
plt.rc('font', family='Bitstream Vera Serif', size=16)
fig = plt.figure(figsize=(8, 6))
axes = plt.axes()

# 2d visualization
VL, VR = np.meshgrid(VL, VR)
plt.contourf(VL ,VR, -I_L, 20,cmap='RdGy')
plt.colorbar(label = ''r'$I/e\Gamma$')
axes.set_xlabel(r'$eV_L$ ')
axes.set_ylabel(r'$eV_R$')

#plt.savefig('test.pdf')
