import nanocavity.operators as no
import nanocavity.franck_condon as nfc
import nanocavity.distributions as nd
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


def G_pm_nm_a(n, m, g_ph, V, kT, DE=0):
    """
    Function to calculate the inelastic tunneling rates for the one level model.
    Parameters:
    -----
    n: int or array like
        photon number of the final state
    m: int or array like
        photon number of the initial state
    g_ph: float
        coupling strength
    V: float
        applied bias
    kT: float
        temperature
    DE: float
        Energy difference between q=1 and q=0

    Returns:
    -----
    G_p_nm_a:    np.array or Float
                The inelastic tunneling rate for transitions from 0 -> 1
    G_m_nm_a:    np.array or Float
                The inelastic tunneling rate for transitions from 1 -> 0
    """
    err_string = (
        "n and m must be nonnegative integers or lists of nonnegative integers."
    )
    for q in (n, m):
        if isinstance(q, (list, np.ndarray)):
            q = np.array(q)
            if q.dtype != np.int64 or (q < 0).any():
                raise TypeError(err_string)
        elif not isinstance(q, int) or q < 0:
            raise TypeError(err_string)

    qf, qi = np.meshgrid(n, m, indexing="ij")
    Fnm = nfc.FC(n, m, g_ph) ** 2
    fp = nd.fermi_dirac(qf - qi - DE, kT, V)
    fm = 1 - nd.fermi_dirac(DE + qi - qf, kT, V)
    if np.isscalar(n) and np.isscalar(m):
        fp = fp[0].item()
        fm = fm[0].item()
    elif np.isscalar(n) or np.isscalar(m):
        fp = fp.ravel()
        fm = fm.ravel()

    G_p_nm = fp * Fnm
    G_m_nm = fm * Fnm
    return G_p_nm, G_m_nm


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


def Pst(g_ph, VL, VR, Gamma_L, Gamma_R, kappa, kT):
    """
    Function that calculates the sationary solution of the population until 2 photons.
    Parameters:
    -----

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

    Returns:
    -----
    P: tuple with pupulations

    """

    Gp00L, Gm00L = G_pm_nm_a(0, 0, g_ph, VL, Gamma_L, kT)
    Gp00R, Gm00R = G_pm_nm_a(0, 0, g_ph, VR, Gamma_R, kT)
    Gp00 = Gamma_L * Gp00L + Gamma_R * Gp00R
    Gm00 = Gamma_L * Gm00L + Gamma_R * Gm00R

    Gp10L, Gm10L = G_pm_nm_a(1, 0, g_ph, VL, Gamma_L, kT)
    Gp10R, Gm10R = G_pm_nm_a(1, 0, g_ph, VR, Gamma_R, kT)
    Gp10 = Gamma_L * Gp10L + Gamma_R * Gp10R
    Gm10 = Gamma_L * Gm10L + Gamma_R * Gm10R

    Gp11L, Gm11L = G_pm_nm_a(1, 1, g_ph, VL, Gamma_L, kT)
    Gp11R, Gm11R = G_pm_nm_a(1, 1, g_ph, VR, Gamma_R, kT)
    Gp11 = Gamma_L * Gp11L + Gamma_R * Gp11R
    Gm11 = Gamma_L * Gm11L + Gamma_R * Gm11R

    Gp12L, Gm12L = G_pm_nm_a(1, 2, g_ph, VL, Gamma_L, kT)
    Gp12R, Gm12R = G_pm_nm_a(1, 2, g_ph, VR, Gamma_R, kT)
    Gp12 = Gamma_L * Gp12L + Gamma_R * Gp12L
    Gm12 = Gamma_L * Gm12R + Gamma_R * Gm12R

    Gp21L, Gm21L = G_pm_nm_a(2, 1, g_ph, VL, Gamma_L, kT)
    Gp21R, Gm21R = G_pm_nm_a(2, 1, g_ph, VR, Gamma_R, kT)
    Gp21 = Gamma_L * Gp21L + Gamma_R * Gp21R
    Gm21 = Gamma_L * Gm21L + Gamma_R * Gm21R

    Gp20L, Gm20L = G_pm_nm_a(2, 0, g_ph, VL, Gamma_L, kT)
    Gp20R, Gm20R = G_pm_nm_a(2, 0, g_ph, VR, Gamma_R, kT)
    Gp20 = Gamma_L * Gp20L + Gamma_R * Gp20R
    Gm20 = Gamma_L * Gm20L + Gamma_R * Gm20R

    Gp22L, Gm22L = G_pm_nm_a(2, 2, g_ph, VL, Gamma_L, kT)
    Gp22R, Gm22R = G_pm_nm_a(2, 2, g_ph, VR, Gamma_R, kT)
    Gp22 = Gamma_L * Gp22L + Gamma_R * Gp22R
    Gm22 = Gamma_L * Gm22L + Gamma_R * Gm22R

    # Eqs 29 - 34 of main text
    Gamma_e0 = Gm00 + Gm10 + Gm20
    Gamma_h0 = Gp00 + Gp10 + Gp20
    P0 = Gamma_e0 / (Gamma_e0 + Gamma_h0)
    Q0 = Gamma_h0 / (Gamma_e0 + Gamma_h0)

    P1 = (
        (Gm10 + Gm20) * Q0 / kappa
        + (Gm11 + Gm21) * (Gp10 + Gp20) * P0 / kappa**2
        + (Gm12 + Gm22) * Gp20 * P0 / (2 * kappa**2)
    )

    Q1 = (
        (Gp10 + Gp20) * P0 / kappa
        + (Gp11 + Gp21) * (Gm10 + Gm20) * Q0 / kappa**2
        + (Gp12 + Gp22) * Gp20 * Q0 / (2 * kappa**2)
    )

    P2 = (
        Gm20 * Q0 / (2 * kappa)
        + Gm21 * (Gp21 + Gp11) * (Gm10 + Gm20) * Q0 / (2 * kappa**3)
        + Gm22 * Gp21 * (Gm10 + Gm20) * Q0 / (4 * kappa**3)
        + Gm21 * (Gp20 + Gp10) * P0 / (2 * kappa**2)
        + Gm22 * Gp20 * P0 / (4 * kappa**2)
    )

    Q2 = (
        Gp21 * (Gm10 + Gm20) * Q0 / (2 * kappa**2)
        + Gp22 * Gp20 * Q0 / (2 * kappa**2)
        + Gp20 * P0 / (2 * kappa)
        + Gp21 * (Gm11 + Gm21) * (Gp10 + Gp20) * P0 / (2 * kappa**3)
        + Gp21 * (Gm12 + Gm22) * Gp20 * P0 / (2 * kappa**3)
        + Gp22 * Gm21 * (Gp20 + Gp10) * P0 / (2 * kappa**3)
        + Gp22 * Gm22 * Gp20 * P0 / (4 * kappa**3)
    )
    return P0, P1, P2, Q0, Q1, Q2
