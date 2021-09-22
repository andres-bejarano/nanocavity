import numpy as np
import matplotlib.pyplot as plt


def fermi_dirac(e, g):
    
    fermi_dirac = 1/(np.exp(e *g)+1)
    
    return fermi_dirac

def current_wide(e,g):
    current_wide =  -((gamma_L(energy)*gamma_R(energy))/\
                   (gamma_L(energy)+gamma_L(energy)))*\
        (fermi_dirac(e, g)-fermi_dirac(e, -g))
    return current_wide

def current(e,g):
    current= -(fermi_dirac(e, g)-fermi_dirac(e, -g))
    return current

def gamma_L(e):
    gamma_L = np.piecewise(e,[e<=0,e>0],\
            [lambda e: 1.-np.sqrt(0.2-e*e),\
             lambda e: 0])
    return gamma_L
def gamma_R(e):
    gamma_R = np.piecewise(e,[e<=0,e>0],\
            [lambda e: 0,\
             lambda e: 0.5+np.sqrt(0.2-e*e)])
    return gamma_R

plt.rc('text', usetex=True)
plt.rc('font', family='Bitstream Vera Serif', size=16)

fig = plt.figure(figsize=(8, 6))
axes = plt.axes()

energy = np.linspace(-1.,1.)
energyl = np.linspace(-1.,0,1001)
energyr = np.linspace(0.,1,1001)


#plt.plot(energy,fermi_dirac(energy, 1000))
#plt.plot(energy,gamma_L(energy))
#plt.plot(energy,gamma_R(energy))



plt.plot(energy,current_wide(energy,1000))
plt.plot(energy,current(energy,1000))
#plt.legend([r'$I_L$', r'$I_R$',r'$I_L+I_R$'])


#axes.set_title(r'c', size=18)
axes.set_xlabel(r'$eV_L/k_BT$ ')
axes.set_ylabel(r'$I/e\Gamma$')

