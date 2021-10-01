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
lg=1.
le=1.
rg=1.
re=1.

#electron_holes = a = 0,1
# 0 if electrons
# 1 if holes
def tunneling_rates(mul,mur,a):
    rate = np.zeros((2,2))
    rate[0,0] = lg * (a-((fermi_dirac(0, mul))*((-1)**(a+1))))
    rate[0,1] = le * (a-((fermi_dirac(delta, mul))*((-1)**(a+1))))
    rate[1,0] = rg * (a-((fermi_dirac(0, mur))*((-1)**(a+1))))
    rate[1,1] = re * (a-((fermi_dirac(delta, mur))*((-1)**(a+1))))
    return rate 

#print(tunneling_rates(1,1,1))  
#here we define:
#gamma_{l,r} matrix for electrons and holes: a
def trans_rate_in(mul,mur):
    gamma_le = np.zeros((4,4))
    gamma_re = np.zeros((4,4))
    gamma_lh = np.zeros((4,4))
    gamma_rh = np.zeros((4,4))
    for i in range(1,3):  
             gamma_le[i,0] = tunneling_rates(mul,mur,0)[0,i-1]
             gamma_le[3,i] = gamma_le[i,0]
             gamma_re[i,0] = tunneling_rates(mul,mur,0)[1,i-1]
             gamma_re[3,i] = gamma_le[i,0]
             #holes
             gamma_lh[0,i] = tunneling_rates(mul,mur,1)[0,i-1]
             gamma_lh[i,3] = gamma_lh[0,i]
             gamma_rh[0,i] = tunneling_rates(mul,mur,1)[1,i-1]
             gamma_rh[i,3] = gamma_lh[0,i]
    return gamma_le, gamma_re ,gamma_lh ,gamma_rh 

#Here a transition rates matrix
def trans_rate(mul,mur):
    trans_rate = np.zeros((4,4)) 
    #we start defining off diagonal elements
    trans_rate = trans_rate_in(mul,mur)[0]+\
         trans_rate_in(mul,mur)[1]+\
           trans_rate_in(mul,mur)[2]+\
               trans_rate_in(mul,mur)[3]      
               #finally diagonal elements    
    for i in range(0,3):
         trans_rate[i,i] = - trans_rate.sum(axis=0)[i]
         #imposing sum of p_i=1
    trans_rate[3]=np.array([1,1,1,1])
    return trans_rate



def populations(mul,mur):
    populations = np.zeros((1,4))
    b = np.array([0, 0,0,1])
    populations = np.linalg.solve(trans_rate(mul, mur),b)
    return populations
#print(populations(1, 1))  


#Finally electronic current
def I(mul,mur):
    I = np.zeros((4,4)) 
    #we start defining off diagonal elements
    x = trans_rate_in(mul,mur)[0]-\
         trans_rate_in(mul,mur)[2] 
    I = np.sum(np.dot(x,populations(mul, mur)))
    return I

print(I(1,0))
#visualization
#plt.rc('text', usetex=True)
#plt.rc('font', family='Bitstream Vera Serif', size=16)
#fig = plt.figure(figsize=(8, 6))
#axes = plt.axes()

#VL = VR = np.linspace(-12, 12, 101)
#VL, VR = np.meshgrid(VL, VR)


# 2d visualization

#plt.contourf(VL ,VR, I(VL,VR),20,cmap='RdGy')
#plt.colorbar(label = ''r'$I/e\Gamma$')
#axes.set_xlabel(r'$eV_L$ ')
#axes.set_ylabel(r'$eV_R$')

    