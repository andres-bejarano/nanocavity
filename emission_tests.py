import numpy as np
import nanocavity.rate_equation as nre
import nanocavity.tls as tls
import matplotlib.pyplot as plt
import nanocavity.distributions as ndist
def peak(height, position, fwhm, wlist):
    # print(wlist.shape)
    # print(position.shape)
    return height * ndist.lorentzian(wlist - position, fwhm)


def parameters(coupling, omegac, delta, kappa, gt):
    detuning = omegac - delta 
    theta = 0.5 * np.arctan(2 * coupling / detuning)
    
    kplusg = kappa * np.cos(theta) ** 2
    wplusg = 4 * gt[0,0]  + kplusg 

    kminusg = kappa * np.sin(theta) ** 2
    wminusg = 4 * gt[0,0] + kminusg 

    return kplusg, wplusg, kminusg,  wminusg

Eg = -0.3
delta = 0.9
omegac = 1
coupling = 0.3

Vt = 3
Vs = -3
kT = 1e-2
gt = 1e-3 * np.eye(2)
gs = 1e-3 * np.eye(2)
kappa = 0.1
wlist = np.linspace(0., 1.8, 10003)
   

H_tls, L, [Nfg, Nfe, Nb] = tls.Hamiltonian('nanocavity', Eg, delta, omegac, coupling, ret_nop=True)

d1 = L[0]
d2 = L[1]
a = L[2]

Een, Est = H_tls.eigsh(k=H_tls.shape[0])

GpL, GmL = nre.transition_rate(Een, Est, [d1, d2], gt, mu=Vt, kT=kT)
GpR, GmR = nre.transition_rate(Een, Est, [d1, d2], gs, mu=Vs, kT=kT)
Kp, Km = nre.transition_rate(Een, Est, [a], kappa, kT=kT, bath='bosonic')
K = Kp + Km
Gp = GpL + GpR
Gm = GmL + GmR 
Gamma = K + Gp + Gm
P = nre.populations(Gamma)
Ire = nre.emission_spectrum(Km, P, Een, wlist, width='full', Kp=Kp, Gp=Gp, Gm=Gm)

E0 = 0
E01 = omegac
Eg1 = Eg + omegac
Ee = Eg + delta
Ee1 = Ee + omegac
Ege = 2 * Eg + delta
Ege1 =  Ege + omegac

rabi =  np.sqrt((omegac - delta) ** 2 + (2 * coupling) ** 2)
Eplus = Eg + (omegac + delta) / 2 + rabi / 2
Eminus = Eg + (omegac + delta) / 2 - rabi / 2  

E = np.array([E0, E01, Eg, Eminus, Eplus, Ee1, Ege, Ege1])
Li = []
diff = np.abs(E[:, np.newaxis] - Een)
Li = np.argmin(diff, axis=1)
[i0, i1, ig, im, ip, ie1, ige, ige1] = Li
kplusg, wplusg, kminusg,  wminusg = parameters(coupling, omegac, delta, kappa, gt) 
I = peak(kminusg * (P[0, im] + P[0, ie1]), Een[im]-Een[ig], wminusg, wlist)
I += peak(kplusg * (P[0, ip] + P[0, ie1]), Een[ip]-Een[ig], wplusg, wlist)
I += peak(kappa * (P[0, i1] + P[0, ige1]), omegac, 4 * gt[0, 0] + kappa, wlist)

plt.figure()
plt.plot(wlist, I, 'k')
plt.plot(wlist, Ire, 'b')
plt.show()
print(np.allclose(I,Ire, atol=1e-4))
