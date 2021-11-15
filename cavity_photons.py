import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import scipy.integrate as integrate

# number of photons
n = 100
b = 0.1

def bose(x):
    return (1./(np.exp(b*x)-1))

#script for stationary solution of populations
x = 1
kup = bose(x)
kdw = bose(x)+1

m = n+1
k = np.zeros((m,m))


k[n,:] = 1.
k[n-1,n] = -kup
k[n-1,n-1] = kdw



for i in range(m-2):
    k[i,i] = kdw*(n-i)
    k[i,i+1] = -kdw*(n-i)-kup*(n+1-i)
    k[i,i+2] = (n-1-i)*kup
    
p = c = np.zeros((m))
c[n] = 1
p = la.solve(k,c)

print(p.sum())


l =  np.linspace(0,n, n+1)


#visualization
plt.rc('text', usetex=True)
plt.rc('font', family='Bitstream Vera Serif', size=16)
fig = plt.figure(figsize=(8, 6))
axes = plt.axes()

axes.set_title(r'Damped cavity: stationary solution for populations', size=18)
axes.set_ylabel(r'$P_n$ ')
axes.set_xlabel(r'$n$ ')


print(bose(l+0.01))
#axes.set_yscale('log')
plt.plot(l, p/p.sum())
plt.plot((l+0.01),bose(l+0.01)*1e-4)
#plt.savefig('test.pdf')

######
plt.legend([r'$P_n$', r'$Bose$'])
#axes.set_title(r'c', size=18)
