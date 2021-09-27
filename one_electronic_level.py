import numpy as np
import matplotlib.pyplot as plt

k=100
# I define fermi dist for electrons = '+' and holes = '-'
def fermi_dirac(e,mu):
    return 1/(np.exp((e-mu)*k)+1)

#tunneling rate as semicircle with wide w 
w=10e0
def gamma(e,mu,intensity):
    gamma = intensity * np.sqrt(1-((e-mu)/w)**2)
    return gamma


#current for asymmetric coupling in which
#Gamma_L = Gamma_R =Gamma/2
# since current invokes gamma we have to add a factor 2.
def current(mul,mur):
    current= -(2*gamma(0,mul,1.)*(gamma(0,mur,1.))/\
                   (gamma(0,mul,1)+gamma(0,mur,1.)))*\
    (fermi_dirac(0,mur)-fermi_dirac(0,mul))
    return current








#visualization
plt.rc('text', usetex=True)
plt.rc('font', family='Bitstream Vera Serif', size=16)

fig = plt.figure(figsize=(8, 6))
axes = plt.axes()

potential = np.linspace(-15.,15.)
x, y = np.meshgrid(potential, potential)
z =  current(x,y)

#plt.plot(chemL,fermi_dirac(0.,chemL,'+'))
#plt.plot(energy,gamma_L(energy))
#plt.plot(energy,gamma_R(energy))


#symmetric chemical potential 
#plt.plot(potential,current(potential,-potential))
#axes.set_xlabel(r'$eV_L$ ')
#axes.set_ylabel(r'$I/e\Gamma$')

# 2d visualization

plt.contourf(x ,y, z,20,cmap='RdGy')
plt.colorbar(label = ''r'$I/e\Gamma$')
axes.set_xlabel(r'$eV_L$ ')
axes.set_ylabel(r'$eV_R$')

######
#plt.legend([r'$I_L$', r'$I_R$',r'$I_L+I_R$'])
#axes.set_title(r'c', size=18)
