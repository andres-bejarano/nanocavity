import numpy as np
import matplotlib.pyplot as plt

#b=1/ k_BT
b=10
# ed= dot energy
ed=0.
def fermi_dirac(ed,mu):
    #I have a problem here
    #error =
    #RuntimeWarning: overflow encountered in exp
     return 1/(np.exp((ed-mu)*b)+1)

#tunneling rate as semicircle with wide w 

def semi_circle(e, mu, w):
    e = np.array(e).reshape(-1, 1)
    mu = np.array(mu).reshape(1, -1)
    x = (e - mu) / w # broadcasting                                                                                                                                                                                
    x = np.clip(x, -1, 1) # values outside the band width should be set to zero
    y = (1 - x ** 2) ** 0.5
    return np.squeeze(y)

#l,r = tunneling rates in the wide band limit 
l=1.
r=l
w=10e1
VL = VR = np.linspace(-10, 10, 101)
gamma_L = l * semi_circle(ed, VL,w)
gamma_R = r * semi_circle(ed, VR,w)

#to have have a current I.shape = (101, 101)
gamma_L = gamma_L.reshape(1, -1)
gamma_R = gamma_R.reshape(-1, 1)

#current for asymmetric coupling in which
#Gamma_L = Gamma_R =Gamma/2
# since current invokes gamma we have to add a factor 2.
fL = fermi_dirac(ed, VL).reshape(1, -1)
fR = fermi_dirac(ed, VR).reshape(-1, 1)

I= -2*(gamma_L * gamma_R / (gamma_L + gamma_R)) *\
    (fR-fL)
    
print('current',I)

#visualization
plt.rc('text', usetex=True)
plt.rc('font', family='Bitstream Vera Serif', size=16)
fig = plt.figure(figsize=(8, 6))
axes = plt.axes()


V = VL.reshape(1, -1) - VR.reshape(-1, 1)
x, y = np.meshgrid(VL, VR)
z = I



#symmetric chemical potential 
#plt.plot(VL,semi_circle(0,VL,1))
#axes.set_xlabel(r'$eV_L$ ')
#axes.set_ylabel(r'$I/e\Gamma$')

# 2d visualization

plt.contourf(V, I,20,cmap='RdGy')
#here also problems
#error =Input z must be 2D, not 0D

#plt.colorbar(label = ''r'$I/e\Gamma$')
#axes.set_xlabel(r'$eV_L$ ')
#axes.set_ylabel(r'$eV_R$')

######
#plt.legend([r'$I_L$', r'$I_R$',r'$I_L+I_R$'])
#axes.set_title(r'c', size=18)
