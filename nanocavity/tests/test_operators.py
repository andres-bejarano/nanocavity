import numpy as np
import nanocavity.operators as no
import nanocavity.qutip.operators as qo
import nanocavity.rate_equation as nre
import qutip as qt



Eg = 0.4
delta = 0.9
omegac = 1
coupling = 0.3

H_parameters = Eg, delta, omegac, coupling

m = 2.5e-2
kappa = 0.1
gL = 1e-3 
gR = 2e-3

VL = 3
VR = -3
kT = 0.1

def test_Htls_nc_QuTiP():
    for rwa in (True, False):
        for n in (1, 2):
            Hnc, [Dg, De, A] = no.H_tls(Eg, delta, omegac, coupling, rwa=rwa, max_bosons=n)
            Hqt, [dg, de, a]  = qo.H_tls(Eg, delta, omegac, coupling, rwa=rwa, max_bosons=n)
            
            Enc, Vnc = Hnc.eigh()
            Eqt, Vqt = Hqt.eigenstates()
            
            Dg = Dg.toarray()
            De = De.toarray()
            A = A.toarray()
    
            dg = dg.full()
            de = de.full()
            a = a.full()
            
            assert np.allclose(Hnc.toarray(), Hqt.full())
            assert np.allclose(Enc, Eqt)
            dim = A.shape[0]
            for i in range(dim):
                M = np.vstack([Vqt[i].full().reshape(dim), Vnc[:, i]])
                rank = np.linalg.matrix_rank(M)
                assert rank == 1

            assert np.allclose(Dg.T @ Dg + Dg @ Dg.T, np.eye(Dg.shape[0]))
            assert np.allclose(Dg.T @ De + De @ Dg.T, 0)
            assert np.allclose(De.T @ Dg + Dg @ De.T, 0)
            assert np.allclose(De.T @ De + De @ De.T, np.eye(De.shape[0]))
            assert np.allclose(A.T @ A - A @ A.T, a.T @ a - a @ a.T)
            
            assert np.allclose(Dg, dg)
            assert np.allclose(De, de)
            assert np.allclose(A, a)
            


def Nanocav(VL=3, VR=-3, kappa=0.1, m=2.5e-2):
    #nanocav-populations
    Hnc, [dg, de, a] = no.H_tls(Eg, delta, omegac, coupling)
    Enc, Vnc = Hnc.eigh()
    GpL, GmL = nre.transition_rate(Enc, Vnc, [dg, de], gL*np.eye(2), mu=VL, kT=kT)
    GpR, GmR = nre.transition_rate(Enc, Vnc, [dg, de], gR*np.eye(2), mu=VR, kT=kT)
    GL = (GpL + GmL)[:, None]  # VL, VR
    GR = (GpR + GmR)[None, :]
    #damping matrix
    Kp, Km = nre.transition_rate(Enc, Vnc, a, kappa, kT=kT, bath='bosonic')
    K = Kp + Km
    #M direct tunneling
    Mp, Mm = nre.bath_system_bath_rate(Enc, Vnc, a, m, VL, VR, kT)
    #transtion rates matrix
    Gamma = K[np.newaxis, np.newaxis] + GL + GR + Mp + Mm
    return nre.populations(Gamma), Kp, Km, GpL, GmL

def test_collapses():
    Pnc, _, _, _, _ = Nanocav()
    _, Hqt, c_ops  = qo.collapses_tls(H_parameters, VL, VR, kappa, gL, gR, kT, m, lead2lead=True, alone=False)
    _, Vqt = Hqt.eigenstates()
    Pqt = qt.steadystate(Hqt.transform(Vqt), list(c_ops)).full().diagonal()
    assert np.allclose(np.sort(Pnc), np.sort(Pqt))


def test_jump_operator():

    for VL in [-1, 1, 2]:
        for VR in [0, -1, -2]:
            Pnc, Kp, Km, GpL, GmL = Nanocav(VL, VR)
            Ig_nc = nre.photo_current(Kp, Km, Pnc)
            Ie_nc = nre.electro_current(GpL - GmL, Pnc)

            
            [dg, de, a], Hqt, c_ops = qo.collapses_tls(H_parameters, VL, VR, kappa, gL, gR, kT, m, lead2lead=True, alone=False)
            _, Vqt = Hqt.eigenstates()

            rho_ss = qt.operator_to_vector(qt.steadystate(Hqt.transform(Vqt), list(c_ops)))


            #left electrode
            cp_gL, cm_gL = qo.collapses(dg, Hqt, kT, bath='fermionic', mu=VL, total=False)
            cp_eL, cm_eL = qo.collapses(de, Hqt, kT, bath='fermionic', mu=VL, total=False)
            CpL = list(np.sqrt(gL) * np.array(cp_gL + cp_eL))
            CmL = list(np.sqrt(gL) * np.array(cm_gL + cm_eL))

            #cavity mode
            cap, cam = qo.collapses(a, Hqt, kT, bath='bosonic', total=False)
            Cp = list(np.sqrt(kappa) * np.array(cap))
            Cm = list(np.sqrt(kappa) * np.array(cam))

            Ig_qt = qt.vector_to_operator((qo.jump(Cm) - qo.jump(Cp)) * rho_ss).tr()   
            Ie_qt = qt.vector_to_operator((qo.jump(CpL) - qo.jump(CmL)) * rho_ss).tr()   

            assert np.allclose(Ig_nc, Ig_qt)
            assert np.allclose(Ie_nc, Ie_qt)


def test_dissipators():
    for kappa in [1e-1, 1]:
        for m in [1e-6, 1e-4, 1e-2]:
            c_ops = list(qo.collapses_tls(H_parameters, VL, VR, kappa, gL, gR, kT, m, lead2lead=True))
            Dnc = qo.dissipator(list(c_ops))
            Dqt = 0
            for c in c_ops:
                Dqt += qt.lindblad_dissipator(c, c)
            assert np.allclose(Dnc.full(), Dqt.full())

def test_liovillian():
    for coupling in [0.005, 0.05, 0.5]:
        Hqt, [dg, de, a] = qo.H_tls(Eg, delta, omegac, coupling)
        Hnc, [Dg, De, A] = no.H_tls(Eg, delta, omegac, coupling)
        _, Vqt = Hqt.eigenstates()
        Enc, Vnc = Hnc.eigh()
        for VL, VR in [[-3, 3], [-1, 3], [0, 3], [1, 3]]:
            c_ops_qt = list(qo.collapses_tls(H_parameters, VL, VR, kappa, gL, gR, kT))
            Lqt = qo.liouvillian(Hqt.transform(Vqt), c_ops_qt)
            
            c_ops_nc = list(no.collapses_tls(H_parameters, VL, VR, kappa, gL, gR, kT))
            Lnc = no.liouvillian(Enc * np.eye(Hnc.shape[0]), c_ops_nc) 
            #not ready
            #assert np.allclose(Lnc, Lqt.full())
