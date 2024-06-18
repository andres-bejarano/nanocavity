import numpy as np
import nanocavity.distributions as ndist
from qutip import qeye, tensor, destroy, sprepost, vector_to_operator, operator_to_vector, spre, spost, fdestroy, sigmaz, lindblad_dissipator

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
                P = Mij * (V[i] * V[j].dag())
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
    return cp, cm

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

                P = (V[i] * V[j].dag())
                c.append(coef * P)
    return c

def jump(c_ops, chi=0):
    J = 0
    for c in c_ops:
                J += sprepost(c, c.dag()) * np.exp(1j * chi)
    return J

def dissipator(c_ops, chi=0, lindblad=False):
    D = 0
    for c in c_ops:
        if lindblad:
            D += lindblad_dissipator(c, c)
        else:
            cdc =  c.dag() * c
            D += sprepost(c, c.dag()) * np.exp(1j * chi) - 0.5 * spre(cdc) - 0.5 * spost(cdc)
    return D

def liouvillian(H, c_ops):
    #incoherent_evolution
    L = 1j * (spre(H) - spost(H))
    L += dissipator(c_ops)
    return L


