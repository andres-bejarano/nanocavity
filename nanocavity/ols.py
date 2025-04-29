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


def collapse_electronic(D, basis, VL, VR, Gamma_L, Gamma_R, kT, total=False, cutoff=0):
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
    total: logical
        Switch wether to return the sum of collapse operators or
        the individual opperators
    cutoff: float
        cutoff for the considered transition matrix elements
    """

    # Electronic collapse operators
    c_pL, c_mL = no.collapses(
        D, basis, kT, "fermionic", Gamma_L, mu=VL, total=False, cutoff=cutoff
    )
    c_pR, c_mR = no.collapses(
        D, basis, kT, "fermionic", Gamma_R, mu=VR, total=False, cutoff=cutoff
    )
    if total:
        return c_pL + c_mL + c_pR + c_mR
    return c_pL, c_mL, c_pR, c_mR


def collapse_dephasing(N_op, basis, kappa, g_ph):
    """
        Function to calculate the dephasing collapse operators for the OLS.

    Parameters:
    -----
    N_op: secondquant operator
        electronic number operator
    basis: list
        (E, V) describing the basis for the collapse operators
    kappa: float
        cavity damping rate
    g_ph: float
        electron-photon coupling strength

    Returns:
    -----
        List of collapse operators for dephasing
    """
    E, V = basis
    M_fi = N_op.inner(V)
    dim = N_op.shape[0]
    c_n = []
    for f in range(dim):
        for i in range(dim):
            P = M_fi[f, i] * V[:, f].reshape(dim, 1) @ V[:, i].reshape(1, dim)
            c_n.append(np.sqrt(kappa * g_ph / 2) * P)
    return c_n


def dissipator_ols(c_ops, method="kron"):
    if not isinstance(c_ops, list):
        raise TypeError("c_ops must be a list")
    dim = c_ops[0].shape[0]
    Id = np.eye(dim)
    # Look https://arxiv.org/pdf/1504.05266
    # Attention: The paper uses column stacking
    # This function uses row stacking
    D = 0
    for c1 in c_ops:
        for c2 in c_ops:
            cdc = c1.conj().T @ c2
            if method == "einsum":
                D += np.einsum("ik,jl->ijkl", c1, c2.conj())
                D -= 0.5 * np.einsum("ik,jl->ijkl", Id, cdc)
                D -= 0.5 * np.einsum("ik,jl->ijkl", cdc.conj(), Id)
            elif method == "kron":
                D += np.kron(c1, c2.conj())
                D -= 0.5 * np.kron(Id, cdc)
                D -= 0.5 * np.kron(cdc.conj(), Id)
    if method == "einsum":
        D = D.reshape((dim**2, dim**2))
    return D


def liouvillian_ols(
    dg, a_ph, Hs, g_ph, VL, VR, Gamma_L, Gamma_R, kappa, kT, cond=False
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

    Returns:
    ----
    L: 2d-np.array
        Liouvillian of the system


    """
    basis = Hs.eigh()
    ng = dg.d * dg
    Dg, A = Lang_Firsov_transform(dg, a_ph, g_ph)
    ca = no.collapses(a_ph, basis, kT, "bosonic", kappa, 0, total=True)
    ce = collapse_electronic(Dg, basis, VL, VR, Gamma_L, Gamma_R, kT, total=True)
    cn = collapse_dephasing(ng, basis, kappa, g_ph)
    c_ops = ce + ca + cn
    L = no.liouvillian(Hs, c_ops, cond=cond)
    return L
