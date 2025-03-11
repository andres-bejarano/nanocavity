import nanocavity.operators as no
import secondquant as sq
import scipy.linalg as sa


def Hamiltonian(hw_ph, E=0, max_bosons=5):
    """
    Function that gives the Hamiltonian for a one level system coupled to a cavity after
    applying the Lang Firsov transformation.

    Parameters:
        -----
        E: float
            energy of the electronic level
        hw_ph: float
            energy of the cavity mode
        g_ph: float
            coupling strength between electrons and photons
        max_bosons: int
            number of photons to be considered

    """
    [dg, a_ph], [ng, n_ph] = sq.composite(1, [max_bosons])
    anni_ops = [dg, a_ph]
    num_ops = [ng, n_ph]
    Hs = E * ng + hw_ph * n_ph + 1e-12 * ng
    return Hs, anni_ops, num_ops


def Lang_Firsov_transform(dg, a_ph, g_ph):
    x = g_ph * (a_ph - a_ph.d)
    X = sq.operator.Operator(sa.expm(x.toarray()))
    Dg = dg * X
    return Dg


def collapse_electronic(Dg, basis, VL, VR, Gamma_L, Gamma_R, kT, total=False, cutoff=0):
    """
        Function to calculate the electronic collapse operators for the OLS.
    Applies a Lang Firsov transform on the electronic operators

    Parameters:
    -----
    Dg: secondquant operator
        electronic annihilation operator AFTER Lang-Firsov transformation
    basis: list
        (E, V) describing the basis for the collapse operators
    VL: float
        bias at left lead
    VR: float
        bias at right lead
    Gamma_L: float
        tunneling rate at the left lead
    Gamma_R: float
        tunneling rate at the right lead
    kT: float
        temperature
    total: logical
        Switch wether to return the sum of collapse operators or
        the individual opperators
    cutoff: float
        cutoff for the considered transition matrix elements
    """

    # Electronic collapse operators
    c_pL, c_mL = no.collapses(
        Dg, basis, kT, "fermionic", Gamma_L, mu=VL, total=False, cutoff=cutoff
    )
    c_pR, c_mR = no.collapses(
        Dg, basis, kT, "fermionic", Gamma_R, mu=VR, total=False, cutoff=cutoff
    )
    if total:
        c_e = c_pL + c_mL + c_pR + c_mR
        return c_e
    return [c_pL, c_mL, c_pR, c_mR]


def collapse_cavity(a_ph, basis, kT, kappa, total=False, cutoff=0):
    cAp, cAm = no.collapses(a_ph, basis, kT, "bosonic", kappa, 0, total=False)
    if total:
        return cAp + cAm
    return cAp, cAm
