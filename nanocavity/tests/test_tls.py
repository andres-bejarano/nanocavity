import qutip as qt
import numpy as np
import nanocavity.master_equation as nme
import nanocavity.tls as tls
import nanocavity.operators as no

#system parameters
Eg = 0.4
Delta = 0.9
hw_ph = 1
g_ph = 0.5
U = 1

H_parameters = Eg, Delta, hw_ph, g_ph

#bath parameters
Gamma_L, Gamma_R = 1e-3, 2e-3
kappa = 0.1
kT = 0.1
 
#the angle of each branch in Jaynes-Cummings model
#\ket{+}_n = \cos{\theta_n}\ket{ng}-i \sin{\theta_n}\ket{n-1,e}
#\ket{-}_n = \sin{\theta_n}\ket{ng}+i \cos{\theta_n}\ket{n-1,e}
def theta(n):
    delta = hw_ph - Delta
    theta = 0.5 * np.arctan(2 * np.sqrt(n) * g_ph / delta)
    return theta

def A_theta():
    tan_theta2 = np.tan(theta(1)) ** 2
    return tan_theta2 +  1 / tan_theta2

#formulas in issue #138
def analytics_sec():
    #huge kappa to have better precision in our analytics
    kappa = 1
    
    x = Gamma_L / Gamma_R
    y =  (Gamma_L + Gamma_R) / (2 * kappa)
    tan2 = np.tan(theta(1)) ** 2
    
    Pg = 2 / (2 + x + 1 / x + A_theta() * y)

    Pplus = tan2 * y * Pg
    Pminus = (1 / tan2) * y * Pg

    P0 = (1 / (2 * x) ) * Pg
    Pge =  (x / 2) * Pg

    return Pg , np.array([P0, Pminus, Pplus, Pge])


#peding to add populations coming from nanocavity.rate_equation()
def test_populations():
    kT= 1e-2
    kappa = 1
    VL, VR = 10, -10
    
    H_parameters = Eg, Delta, hw_ph, g_ph, U
    H, [dg, de, a] = tls.Hamiltonian('nanocavity', *H_parameters)
    rho = tls.rho_st('nanocavity', H_parameters, VL, VR, kappa, Gamma_L, Gamma_R, kT)

    #rho is written in the basis without interaction
    E, V = H.eigh()    
    Vinv = np.linalg.inv(V)
    rho = Vinv @ rho @ V
    Pme = rho.diagonal().real
    
    Pgme, Pme = Pme[1], [Pme[0], Pme[2], Pme[4] , Pme[6]]
    Pgan, Pan = analytics_sec()
    assert np.allclose(Pgme, Pgan, atol=1e-2)
    assert np.allclose(Pme, Pan, atol=1e-3)


def test_stationary():
    for VL, VR in [[3, -3], [2, 0], [-1, 2]]:
        for g_ph in [0.005, 0.05, 0.5]:
            H_parameters = Eg, Delta, hw_ph, g_ph, U
            H, [Dg, De, A] = tls.Hamiltonian('nanocavity', *H_parameters)
            Pme = tls.rho_st('nanocavity', H_parameters, VL, VR, kappa, Gamma_L, Gamma_R, kT)
            Pqt = tls.rho_st('qutip', H_parameters, VL, VR, kappa, Gamma_L, Gamma_R, kT)
            assert np.allclose(Pme, Pqt)

def test_correlation_AB():
    tlist = np.linspace(0., 200, 101)
    for g_ph in [0.005, 0.05, 0.5]:
        H_parameters = Eg, Delta, hw_ph, g_ph, U
        Hnc, [_, _, A] = tls.Hamiltonian('nanocavity', *H_parameters)
        Hqt, [_, _, a] = tls.Hamiltonian('qutip', *H_parameters)
        VL, VR = 3, -3
        for iva in (False, True):
            c_nc = tls.collapses('nanocavity', H_parameters, VL, VR, kappa, Gamma_L, Gamma_R, kT, iva=iva)
            L = no.liouvillian(Hnc, list(c_nc))
            Snc = nme.correlation_AB(L, A.d, A, tlist)

            c_qt = tls.collapses('qutip', H_parameters, VL, VR, kappa, Gamma_L, Gamma_R, kT, iva=iva)
            rho_st= qt.steadystate(Hqt, list(c_qt))
            Sqt = qt.correlation_2op_1t(H=Hqt, state0=rho_st, taulist=tlist, c_ops=list(c_qt), a_op=a.dag(), b_op=a)
            assert np.allclose(Snc.real, Sqt.real, atol=1e-5)
            assert np.allclose(Snc.imag, Sqt.imag, atol=1e-5)



def test_spectrum():
    wlist = np.linspace(0., 1.8, 103)
    Inc = tls.spectrum('nanocavity', H_parameters, 3, -3, kappa, Gamma_L, Gamma_R, kT, wlist)
    Iqt = tls.spectrum('qutip', H_parameters, 3, -3, kappa, Gamma_L, Gamma_R, kT, wlist)
    assert np.allclose(Inc, Iqt)



"""peding to develop
def test_current():
    for VL, VR in [[3, -3], [2, 0], [-1, 2]]:
            [dg, de, a], Hqt, c_ops = tls.collapses('qutip', H_parameters, VL, VR, kappa, gL, gR, kT, alone=False)

            #left electrode
            cp_gL, cm_gL = qo.collapses( dg, Hqt, kT, bath='fermionic', mu=VL, total=False)
            cp_eL, cm_eL = qo.collapses(de, Hqt, kT, bath='fermionic', mu=VL, total=False)
            CpL = list(np.sqrt(gL) * np.array(cp_gL + cp_eL))
            CmL = list(np.sqrt(gL) * np.array(cm_gL + cm_eL))

            #cavity mode
            cap, cam = qo.collapses(a, Hqt, kT, bath='bosonic', total=False)
            Cp = list(np.sqrt(kappa) * np.array(cap))
            Cm = list(np.sqrt(kappa) * np.array(cam))

            L = qo.liouvillian(Hqt, list(c_ops))

            Ig_me = qme.current(qo.jump(Cm) - qo.jump(Cp), L)
            Ie_me = qme.current(qo.jump(CpL) - qo.jump(CmL), L)
            Ig_re, Ie_re, _, _ = fcs(VL, VR)

            #assert np.allclose(Ig_me, Ig_re)
            #assert np.allclose(Ie_me, Ie_re)"""

def test_g2():
    VL, VR = 10, -10
    g_ph = 0.005
    Delta = 0.99
    H_parameters = Eg, Delta, hw_ph, g_ph, U
    tlist = np.linspace(0., 300, 10)
    g2nc = tls.g2('nanocavity', H_parameters, VL, VR, kappa, Gamma_L, Gamma_R, kT, tlist)
    g2qt = tls.g2('qutip', H_parameters, VL, VR, kappa, Gamma_L, Gamma_R, kT, tlist)
    assert np.allclose(g2nc, g2qt, atol=1e-1)

