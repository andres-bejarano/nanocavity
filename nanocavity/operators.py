import numpy as np
import nanocavity.distributions as ndist
import secondquant as sq
from qutip import qeye, tensor, destroy, sprepost, vector_to_operator, operator_to_vector, spre, spost, fdestroy, sigmaz

#two level system coupled to single cavity mode
def H_tls_nc(Eg, delta, omega, coupling, rwa=True, max_bosons=1, ret_nop=False):
    [dg, de, a], [Nfg, Nfe, Nb] = \
        sq.composite(fermion_modes=2, boson_modes=1, max_bosons=max_bosons)
    H0 = Eg * Nfg + (Eg +  delta) * Nfe + omega * Nb
    if rwa:
        Hint = coupling * (a.d * dg.d * de + a * de.d * dg)
    else:
        Hint = coupling * (a + a.d) * (dg.d * de + de.d * dg)
    H = H0 +  Hint
    L = [dg, de, a]
    if ret_nop:
        return H, L, [Nfg, Nfe, Nb]
    return H, L

#two level system coupled to single cavity mode in QuTiP

def H_tls_QuTiP(Eg, delta, omega, coupling, rwa=True, max_bosons=1):
    N = max_bosons + 1
    dg = tensor(fdestroy(2, 0), qeye(N))
    de = tensor(fdestroy(2, 1), qeye(N))
    
    #sigmaz = [[1, 0], [0, -1]] is playing the role of permutation as
    # |n_g, n_e> = -|n_e, n_g>
    a = tensor(sigmaz(), sigmaz(), destroy(N))
    
    H0 = Eg * dg.dag() * dg + (Eg + delta)* de.dag() * de + omega * a.dag() * a
    if rwa:
        Hint = coupling * (a.dag() * dg.dag() * de + a * de.dag() * dg)
    else:
        Hint = coupling * (a + a.dag()) * (dg.dag() * de + de.dag() * dg)
    H = H0 + Hint
    E, V = H.eigenstates()
    L = [dg, de, a]
    return H, L


#collapses operators

def collapses(A_op, H, kT, bath, mu=0, total=True, cutoff=1e-12):
    """
    This script describes the collapse operator defined below. We will have an operator A_op of the system that interacts with a given bath, which is in thermal equilibrium and will be characterized by the Fermi-Dirac or Bose-Einstein function depending on its nature.

    To identify the collapse operators, we must write the dissipator of our problem 
        \\mathcal{D}^+[\\rho] = \\sum_{ij}  dist+(E_{ji}) A_{ji}^\\dagger\\rho A_{ij} - \\frac{1}{2}\\{A_{iJ} A^\\dagger_{ji}, \\rho\\}

         \\mathcal{D}^-[\\rho] = \\sum_{ij}  dist^-(E_{ji}) A_{ij}\\rho A_{ji}^\dagger - \\frac{1}{2}\\{A_{ji}^\\dagger A_{ij}, \\rho\\}


    where A_{ij} is a system eigenoperator

    A_ij = \\langle i \\rvert A \\lvert j \\rangle \\lvert i \\rangle \\langle j \\rvert   

    the collapse is defined as:

    C^+ = \\sqrt{dist^+(E_{ji}) } \\lvert \\langle i \\rvert A^\dagger \\lvert j \\rangle\\rvert^2 \\lvert i \\rangle \\langle j \\rvert

    C^- = \\sqrt{dist^-(E_{ji}) } \\lvert \\langle i \\rvert a \\lvert j \\rangle\\rvert^2 \\lvert i \\rangle \\langle j \\rvert

    Parameters
    ----------
    A_op : Qobj or QobjEvo
           Annihilation operator

    H : Qobj
        system hamiltonian.

    kT : float
         Temperature of the bath.

    bath :  str
         fermionic or bosonic bath.


    Returns
    -------
    cp, cm : List of Qobj
        Two list of operators for adding or removing particles.
    """
    #We can place this line outside of this function, but for now, we can accept this overhead since it makes the code easier to read and does not impose a significant cost, as we call the collapse function at most twice.
    E, V = H.eigenstates()
    cp, cm = [], []
    for i, Ei in enumerate(E):
        for j, Ej in enumerate(E):
            Mij = A_op.matrix_element(V[i], V[j]) 
            if abs(Mij) > cutoff:
                Eji = Ej - Ei
                P = Mij * (V[i] * V[j].dag()).transform(V)
                if bath=='bosonic':
                    nb = ndist.bose_einstein(Eji, kT=kT)
                    cp.append(np.sqrt(nb) * P.dag())
                    cm.append(np.sqrt(1 + nb) * P)
                elif bath=='fermionic':
                    fd = ndist.fermi_dirac(Eji, kT=kT, mu=mu)
                    cp.append(np.sqrt(fd) * P.dag())
                    cm.append(np.sqrt(1 - fd) * P)
    if total:
        return cp + cm
    else:
        return cp, cm


def collapses_tls_QuTiP(H_parameters, VL, VR, kappa, gL, gR, kT, m=0, lead2lead=False, alone=True):
    [Eg, delta, omegac, coupling] = H_parameters
    H, [dg, de, a] = H_tls_QuTiP(Eg, delta, omegac, coupling)
    
    #left electrode
    c_gL = collapses(dg, H, kT, bath='fermionic', mu=VL)
    c_eL = collapses(de, H, kT, bath='fermionic', mu=VL)
    CL = np.sqrt(gL) * np.array(c_gL + c_eL)

    #right electrode
    c_gR = collapses(dg, H, kT, bath='fermionic', mu=VR)
    c_eR = collapses(de, H, kT, bath='fermionic', mu=VR)
    CR = np.sqrt(gR) * np.array(c_gR + c_eR)

    #cavity mode
    CA = collapses(a, H, kT, bath='bosonic')

    CA = np.sqrt(kappa) * np.array(CA)
    
    if lead2lead:
        c_lead = lead_cavity_lead_collapses(a, H, VL, VR, kT, m)
        c_ops = np.concatenate((CL, CR, CA, c_lead))
    else:
        c_ops = np.concatenate((CL, CR, CA))
    
    if alone:
        return c_ops
    else:
        S_op =  [dg, de, a] 
        return S_op, H, c_ops


def lead_cavity_lead_collapses(A_op, H, VL, VR, kT, m):
    """
    This collapses represent the tunneling electron from left/right to right/left electrode interacting with the cavity mode.

    To identify the collapse operators, we must write the dissipator of our problem 
        \\mathcal{L}^+[\\rho] = \\sum_{ij} M F(E_{ji}) a_{ij}^\\dagger\\rho a_{ji} - \\frac{1}{2}\\{a_{ij} a^\\dagger_{ji}, \\rho\\}

         \\mathcal{L}^-[\\rho] = \\sum_{ij} M F(E_{ji}) a_{ij}^\\dagger\\rho a_{ji} - \\frac{1}{2}\\{a_{ij} a^\\dagger_{ji}, \\rho\\}


    where a_{ij} is a cavity system eigenoperator

    a_ij = \\langle i \\rvert a \\lvert j \\rangle \\lvert i \\rangle \\langle j \\rvert   

    the collapse is defined as:

    C^+ = \\sqrt{M F(E_{ji}) } \\lvert \\langle i \\rvert a^\dagger \\lvert j \\rangle\\rvert^2 \\lvert i \\rangle \\langle j \\rvert

    C^- = \\sqrt{M F(E_{ji}) } \\lvert \\langle i \\rvert a \\lvert j \\rangle\\rvert^2 \\lvert i \\rangle \\langle j \\rvert

    
    where F(x) = \\sum_{s=L,R}\\frac{V_{s} - V_{\\bar{s}} +  x}{1-e^{-(V_{s} - V_{\\bar{s}} +  x)/k_BT}}

    Parameters
    ----------
    A_op : Qobj or QobjEvo
           Annihilation operator for bosons.

    E, V : Qobj
           Eigenvalues and eigenstates of system hamiltonian.

    VL, VR: float
            Left/Right chemical potential (VL, VR).

    kT: float
        Temperature of the bath

    M: float
       Coupling between leads


    Returns
    -------
    D : List of Qobj
        Collpases operators.
    """

    E, V = H.eigenstates()
    c = []
    for i, Ei in enumerate(E):
        for j, Ej in enumerate(E):
            #As the distribution functions is the same for both dissipator
            #we can write at the same time both collapses. 
            #Each time that we create o remove a photon 
            #we have no zero value for Mij and then we create the collapse.
            Mij = (A_op + A_op.dag()).matrix_element(V[i], V[j]) ** 2
            if Mij != 0:
                Eji = Ej - Ei
                VLR = VL - VR
    
                F1 = ndist.Fermi_cb(VLR+Eji, kT)
                F2 = ndist.Fermi_cb(-VLR+Eji, kT)
                coef = np.sqrt(m * (F1 + F2) * Mij)
    
                P = (V[i] * V[j].dag()).transform(V)
                c.append(coef * P)
    return c


def jump_operator(A_op, E, V, distribution):
    J = 0
    for i, Ei in enumerate(E):
        for j, Ej in enumerate(E):
            Mij = A_op.matrix_element(V[i], V[j])
            if Mij != 0:
                aij = Mij * (V[i] * V[j].dag()).transform(V)
                J += distribution(Ej - Ei) * sprepost(aij, aij.dag()) 
    return J



def jump_bosonic(A_op, E, V, kT, rate='in'):
    dist = ndist.bath_dist(E, kT, rate, bath='bosonic')
    return jump_operator(A_op, E, V, dist)

def jump_fermionic(A_op, E, V, mu, kT, rate='in'):
    dist = ndist.bath_dist(E, kT, rate, bath='fermionic', mu=mu)
    return jump_operator(A_op, E, V, dist)

def jump_lead(A_op, E, V, eV, kT, rate='in'):
    dist = ndist.bath_dist(E, kT, rate, bath='leadtolead', mu=0, eV=eV)
    return jum_operator(A_op, E, V, dist)

def dissipator(A_op, E, V, distribution, chi):
    L = 0
    for i, Ei in enumerate(E):
        for j, Ej in enumerate(E):
            Mij = A_op.matrix_element(V[i], V[j])
            if Mij != 0:
                Eji = Ej - Ei
                aij = Mij * (V[i] * V[j].dag()).transform(V)
                aca =  aij.dag() * aij
                L += distribution(Eji) * (sprepost(aij, aij.dag()) * np.exp(1j * chi) - \
                        0.5 * spre(aca) - 0.5 * spost(aca)) 
    return L

def dissipator_bosonic(A_op, E, V, kT, rate, chi=0):
    dist = ndist.bath_dist(E, kT, rate, bath='bosonic')
    return dissipator(A_op, E, V, dist, chi)

def dissipator_fermionic(A_op, E, V, mu, kT, rate, chi=0):
    dist = ndist.bath_dist(E, kT, rate, bath='fermionic', mu=mu)
    return dissipator(A_op, E, V, dist, chi)

def dissipator_lead(A_op, E, V, eV, kT, rate, chi=0):
    dist = ndist.bath_dist(E, kT, rate, bath='leadtolead', mu=0, eV=eV)
    return dissipator(A_op, E, V, dist, chi)

def Liouvillian(H, S_op, VL, VR, kT=1e-2, kappa=0.1, gL=1e-3, gR=1e-3, m=0, iva=False, chi_b=0, chi_f=0):
    [dg, de, a] = S_op
    
    if iva:
        Hint = coupling * (a.dag() * dg.dag() * de + a * de.dag() * dg)
        H -= Hint
    E, V = H.eigenstates()
    
    #cavity-radiation_bath dissipator
    L = kappa * (dissipator_bosonic(a, E, V, kT, rate='out', chi=chi_b) + \
                 dissipator_bosonic(a.dag(), E, V, kT, rate='in', chi=-chi_b))

    #molecule-leads dissipator
    L += gL * (dissipator_fermionic(dg, E, V, VL, kT, rate='out', chi=chi_f) + \
               dissipator_fermionic(de, E, V, VL, kT, rate='out', chi=chi_f) + \
               dissipator_fermionic(dg.dag(), E, V, VL, kT, rate='in', chi=-chi_f) + \
               dissipator_fermionic(de.dag(), E, V, VL, kT, rate='in', chi=-chi_f))

    L += gR * (dissipator_fermionic(dg, E, V, VR, kT, rate='out') + \
               dissipator_fermionic(de, E, V, VR, kT, rate='out') + \
               dissipator_fermionic(dg.dag(), E, V, VR, kT, rate='in') + \
               dissipator_fermionic(de.dag(), E, V, VR, kT, rate='in'))
    
    #cavity-leads dissipator
    VLR = VL - VR
    VRL = VR - VL
    L += m * (dissipator_lead(a, E, V, eV=VLR, kT=kT, rate='out', chi=chi_f) + \
              dissipator_lead(a.dag(), E, V, eV=VLR, kT=kT, rate='in', chi=chi_f) + \
              dissipator_lead(a, E, V, eV=VRL, kT=kT, rate='out', chi=-chi_f) + \
              dissipator_lead(a.dag(), E, V, eV=VRL, kT=kT, rate='in', chi=-chi_f))
    if iva:
        H += Hint

    #incoherent_evolution 
    L += -1.0j * (spre(H.transform(V)) - spost(H.transform(V)))
    return L




