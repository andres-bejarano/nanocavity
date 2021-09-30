import numpy as np
import matplotlib.pyplot as plt

#b=1/ k_BT
b=10
# ed= dot energy
ed=0.
def fermi_dirac(ed,mu):
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
VL = VR = np.linspace(-1, 1, 101)
gamma_L = l * semi_circle(ed, VL,w)
gamma_R = r * semi_circle(ed, VR,w)

#to have have a current I.shape = (101, 101)
gamma_L = gamma_L.reshape(1, -1)
gamma_R = gamma_R.reshape(-1, 1)
fL = fermi_dirac(ed, VL).reshape(1, -1)
fR = fermi_dirac(ed, VR).reshape(-1, 1)


#current for asymmetric coupling in which
#Gamma_L = Gamma_R =Gamma/2
# since current invokes gamma we have to add a factor 2.
I= -2*(gamma_L * gamma_R / (gamma_L + gamma_R)) *\
    (fR-fL)
    


#visualization
plt.rc('text', usetex=True)
plt.rc('font', family='Bitstream Vera Serif', size=16)
fig = plt.figure(figsize=(8, 6))
axes = plt.axes()



# 2d visualization

VL, VR = np.meshgrid(VL, VR)
plt.contourf(VL, VR, I,20,cmap='RdGy')


plt.colorbar(label = ''r'$I/e\Gamma$')
axes.set_xlabel(r'$eV_L$ ')
axes.set_ylabel(r'$eV_R$')

plt.savefig('test.pdf')

######
#plt.legend([r'$I_L$', r'$I_R$',r'$I_L+I_R$'])
#axes.set_title(r'c', size=18)
