import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import scipy.integrate as integrate

#b=1/ k_BT
b=10

def fermi_dirac(e,mu):
    return 1/(np.exp((e-mu)*b)+1)
 
# levels
nlev=2.
#(0,e,g,2)=(0,1,2,3)
ed=0.
delta=1.
e = np.array([ed,delta])
#here tunneling rates for \Gamma_{Le,lg,re,rg}
lg=1.
le=1.
rg=1.
re=1.
tun_rates = np.array([[lg,le],[rg,re]])

#electron_holes = a = 0,1
# 0 if electrons
# 1 if holes
#pending to improve this function
def tunneling_rates(mul,mur,a):
    rate = np.zeros((2,2))
    rate[0,0] = tun_rates[0,0] * (a-((fermi_dirac(0, mul))*((-1)**(a+1))))
    rate[0,1] = tun_rates[0,1] * (a-((fermi_dirac(delta, mul))*((-1)**(a+1))))
    rate[1,0] = tun_rates[1,0] * (a-((fermi_dirac(0, mur))*((-1)**(a+1))))
    rate[1,1] = tun_rates[1,1] * (a-((fermi_dirac(delta, mur))*((-1)**(a+1))))
    return rate 

print(tunneling_rates(10,10,0))  
#here we define:
#gamma_{l,r} matrix for electrons and holes: a
def trans_rate_in(mul,mur,a):
         gamma_l = np.zeros((4,4))
         gamma_r = gamma_l
         for i in range(1,3):  
             gamma_l[i,0] = tunneling_rates(mul,mur,a)[0,i-1]
             gamma_l[3,i-1] = gamma_l[i-1,0]
             gamma_r[i,0] = tunneling_rates(mul,mur,a)[1,i-1]
             gamma_r[3,i-1] = gamma_l[i-1,0]
             i=i+1
             return gamma_l, gamma_r

        
#Here a transition rates matrix
def trans_rate(mul,mur):
     trans_rate = np.zeros((4,4)) 
     #we start defining off diagonal elements
     trans_rate = trans_rate_in(mul,mur,1)[0]+\
         trans_rate_in(mul,mur,1)[0]+\
           trans_rate_in(mul,mur,0)[0]+\
               trans_rate_in(mul,mur,0)[1]      
     #finally diagonal elements    
     for i in range(0,3):
         trans_rate[i,i] = - trans_rate.sum(axis=0)[i]
         #imposing sum of x_i=1
         trans_rate[3]=np.array([1,1,1,1])
         return trans_rate




def populations(mul,mur):
    b = np.array([0, 0,0,1])
    populations = np.linalg.solve(trans_rate(mul, mur),b)
    return populations
print(populations(1., -1.))
   
#Finally electronic current

def current(mul,mur):
    gamma = trans_rate_in(mul,mur,1)[0]-trans_rate_in(mul,mur,0)[0]
    current = np.dot(gamma,populations(mul,mur))
    return current



#visualization
#plt.rc('text', usetex=True)
#plt.rc('font', family='Bitstream Vera Serif', size=16)

#fig = plt.figure(figsize=(8, 6))
#axes = plt.axes()

#potential = np.linspace(-15.,15.)
#x, y = np.meshgrid(potential, potential)
#z =  current(x,y)

# 2d visualization

#plt.contourf(x ,y, z,20,cmap='RdGy')
#plt.colorbar(label = ''r'$I/e\Gamma$')
#axes.set_xlabel(r'$eV_L$ ')
#axes.set_ylabel(r'$eV_R$')

    