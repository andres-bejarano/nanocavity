import numpy as np
import nanocavity.distributions as ndist
from qutip import sprepost, spre, spost, lindblad_dissipator


def collapses(A_op, H, kT, bath, rate, mu=0, total=True, cutoff=1e-12):
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

    rate: float
        coupling strength


    Returns
    -------
    cp, cm : List of Qobj
        Two list of operators for adding or removing particles.
    """
    # We can place this line outside of this function, but for now, we can accept this overhead since it makes the code easier to read and does not impose a significant cost, as we call the collapse function at most twice.
    E, V = H.eigenstates()
    dim = E.shape[0]
    E_fi = E.reshape(dim, 1) - E.reshape(1, dim)
    if bath == "bosonic":
        # This rate is for photon absorption, thus the final state must be
        # higher in energy than the initial one
        nb_fi_p = np.where(E_fi > 0, ndist.bose_einstein(E_fi, kT), 0)
        # This rate is for photon emission, thus the final state must be
        # lower in energy than the initial one
        nb_fi_m = np.where(E_fi < 0, 1 + ndist.bose_einstein(-E_fi, kT), 0)
    elif bath == "fermionic":
        fd_fi_p = ndist.fermi_dirac(E_fi, kT, mu)
        fd_fi_m = 1 - ndist.fermi_dirac(-E_fi, kT, mu)

    cp, cm = [], []
    for f in range(dim):
        for i in range(dim):
            Mfi = A_op.matrix_element(V[f], V[i])
            if abs(Mfi) > cutoff:
                P = Mfi * (V[f] * V[i].dag())
                if bath == "bosonic":
                    cp.append(np.sqrt(rate * nb_fi_p.T[f, i]) * P.dag())
                    cm.append(np.sqrt(rate * nb_fi_m[f, i]) * P)
                elif bath == "fermionic":
                    cp.append(np.sqrt(rate * fd_fi_p.T[f, i]) * P.dag())
                    cm.append(np.sqrt(rate * fd_fi_m[f, i]) * P)
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
            # As the distribution functions is the same for both dissipator
            # we can write at the same time both collapses.
            # Each time that we create o remove a photon
            # we have no zero value for Mij and then we create the collapse.
            Mij = (A_op + A_op.dag()).matrix_element(V[i], V[j]) ** 2
            if Mij != 0:
                Eji = Ej - Ei
                VLR = VL - VR

                F1 = ndist.Fermi_cb(VLR + Eji, kT)
                F2 = ndist.Fermi_cb(-VLR + Eji, kT)
                coef = np.sqrt(m * (F1 + F2) * Mij)

                P = V[i] * V[j].dag()
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
            cdc = c.dag() * c
            D += (
                sprepost(c, c.dag()) * np.exp(1j * chi)
                - 0.5 * spre(cdc)
                - 0.5 * spost(cdc)
            )
    return D


def liouvillian(H, c_ops):
    # incoherent_evolution
    L = 1j * (spre(H) - spost(H))
    L += dissipator(c_ops)
    return L
