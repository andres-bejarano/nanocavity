import numpy as np
import matplotlib.pyplot as plt

#k = k_BT
k=100
# ed= dot energy
def fermi_dirac(ed,mu):
    return 1/(np.exp((ed-mu)*k)+1)

#tunneling rate as semicircle with wide w 
w=10e1
ed=0.
def gamma(mu,intensity,ef):
#attempt to insert a dependence of fermi energy as step function 
    gamma = np.piecewise(mu, [mu+ef<ed,mu+ef>=ed],\
                [lambda mu: 0,\
                 lambda mu: intensity * np.sqrt(1-((ed-mu)/w)**2)])
    return gamma

# ef_{alpha}: fermy energy in the \alpha electrode
efl = 100
efr = efl
#l,r = tunneling rates in the wide band limit 
l=1.
r=l
#current for asymmetric coupling in which
#Gamma_L = Gamma_R =Gamma/2
# since current invokes gamma we have to add a factor 2.
def current(mul,mur):
    current= -(2*gamma(mul,l,efl)*(gamma(mur,r,efr))/\
                   (gamma(mul,l,efl)+gamma(mur,r,efr)))*\
    (fermi_dirac(ed,mur)-fermi_dirac(ed,mul))
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
