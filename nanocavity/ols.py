import nanocavity.operators as no
import secondquant as sq
from secondquant.operator import Operator
import scipy.linalg as sa
import numpy as np


def Hamiltonian(hw_ph, E=1e-12, max_bosons=5):
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
    [d, a_ph], [ng, n_ph] = sq.composite(1, [max_bosons])
    anni_ops = [d, a_ph]
    num_ops = [ng, n_ph]
    Hs = E * ng + hw_ph * n_ph
    return Hs, anni_ops, num_ops


def Lang_Firsov_transform(d, a_ph, g_ph):
    """
    Function calculating the Lang-Firsov transform of an electronic annihilation operator
    Parameters:
        ----
        d: secondquant operator
            electronic annihilation operator
        a_ph: secondquant operator
            bosonic annihilation operator
        g_ph: float
            coupling strength between electrons and bosons

    Returns:
        ---
        D: secondquant operator
            Lang-Firsov transformed electronic annihilation operator
        A_ph: secondquant operator
            Shifted bosonice operator

    """
    x = g_ph * (a_ph - a_ph.d)
    X = sq.operator.Operator(sa.expm(x.toarray()))
    D = d * X
    A = a_ph - g_ph * d.d * d
    return D, A


def collapse_electronic(D, basis, VL, VR, Gamma_L, Gamma_R, kT, cutoff=0):
    """
        Function to calculate the electronic collapse operators for the OLS.

    Parameters:
    -----
    D: secondquant operator
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
    cutoff: float
        cutoff for the considered transition matrix elements
    """

    bin_width = 1e-6
    # Electronic collapse operators
    c_pL, c_mL = no.collapses(
        D, basis, kT, "fermionic", Gamma_L, bin_width, mu=VL, cutoff=cutoff
    )
    c_pR, c_mR = no.collapses(
        D, basis, kT, "fermionic", Gamma_R, bin_width, mu=VR, cutoff=cutoff
    )
    return [c_pL, c_pR], [c_mL, c_mR]


def liouvillian(
    dg, a_ph, Hs, g_ph, VL, VR, Gamma_L, Gamma_R, kappa, kT, cond=False, method="kron"
):
    """
        Function to build the Liouvillian for the one level system coupled to a cavity

    Parameters:
    ----
    dg: secondquant operator
        electronic annihilation operator Before Lang-Firsov transformation
    a_ph: secondquant operator
        photonic annihilation operator
    Hs: secondquant operator
        Hamiltonian of the system
    g_ph: float
        electron-photon coupling strength
    VL: float
        bias at left lead
    VR: float
        bias at right lead
    Gamma_L: float
        tunneling rate at the left lead
    Gamma_R: float
        tunneling rate at the right lead
    kappa: float
        cavity damping rate
    kT: float
        temperature
    cond: logical
        If True show the condition number of the Liouvillian
        Defaults to False
    method: string
        method on how to build the Liouvillian
        defaults to "kron"

    Returns:
    ----
    L: 2d-np.array
        Liouvillian of the system


    """
    basis = Hs.eigh()
    ng = dg.d * dg
    Dg, A = Lang_Firsov_transform(dg, a_ph, g_ph)
    ca = {"full": [np.sqrt(kappa) * a_ph.toarray()]}
    [c_pL, c_pR], [c_mL, c_mR] = collapse_electronic(
        Dg, basis, VL, VR, Gamma_L, Gamma_R, kT
    )
    cn = {"full": [np.sqrt(kappa * g_ph**2 / 2) * ng.toarray()]}
    cops = [c_pL, c_pR, c_mL, c_mR, cn, ca]

    L = no.liouvillian(Hs, cops, method=method, cond=cond)

    return L

def get_diagonal_indices(max_bosons, cutoff=None):
    """
    Returns diagonal indices of Hs for a truncated photon basis in each charge sector q = 0 and q = 1.

    Parameters:
    - max_bosons (int): Maximum photon number used to construct Hs (sets size of blocks)
    - cutoff_q0 (int or None): Max photon number to keep for q=0, 1 (inclusive). If None, use full range (0 to max_bosons)

    Returns:
    - List of [i, i] indices for the truncated diagonal
    """
    indices = []

    if cutoff is None:
        cutoff = max_bosons

    # Block q = 0
    for n in range(cutoff + 1):
        indices.append([n, n])

    # Block q = 1
    offset = max_bosons + 1
    for n in range(cutoff + 1):
        indices.append([offset + n, offset + n])

    return indices



def get_basis_index(n, q, max_bosons):
    """
    Returns the index in the full Hilbert space basis for a state |n, q⟩
    """
    if q == 0:
        return n
    elif q == 1:
        return max_bosons + 1 + n
    else:
        raise ValueError("q must be 0 or 1")

def get_vectorized_rho_index(n1, q1, n2, q2, max_bosons):
    """
    Returns the index in the vectorized density matrix vec(rho) for ⟨n1,q1| ρ |n2,q2⟩
    with row-stacking (vec(rho)[i,j] = i * d + j)
    """
    # Dimension of total Hilbert space
    d = 2 * (max_bosons + 1)

    # Get row (bra) and column (ket) indices
    i = get_basis_index(n1, q1, max_bosons)  # bra
    j = get_basis_index(n2, q2, max_bosons)  # ket

    # Row-stacked vectorized index
    return i * d + j
