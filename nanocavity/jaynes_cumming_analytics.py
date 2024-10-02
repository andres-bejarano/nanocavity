import numpy as np
import nanocavity.operators as no
import nanocavity.master_equation as nme
import nanocavity.tls as tls
import nanocavity.distributions as ndist


"""
Generalized Jaynes-Cummings model:

    H = E_g d_g^\dagger d_g + (E_g + \Delta) d_e^\dagger d_e + U d_g^\dagger * d_e^\dagger * d_e * d_g + hw_ph * a^\dagger a + g_ph(a^\dagger * d_g^\dagger d_e + a d_e^\dagger d_g)

The interaction mix sttes \ket{ng} and \ket{n-1, g}

    \ket{+}_n = \cos{\theta_n}\ket{ng}-i \sin{\theta_n}\ket{n-1,e}
    \ket{-}_n = \sin{\theta_n}\ket{ng}+i \cos{\theta_n}\ket{n-1,e}

the angle \theta_n is defined as:
    
    \tan 2 \theta_n = \frac{2\sqrt(n)g_ph}{delta} 
    
where delta = hw_ph - Delta the detuning.

The energies are given by:

    E_n^{\pm} = Eg + \frac{\Delta + (2n-1)hw_ph}{2} \pm \Omega_R^n


where the Rabi frequency (\Omega_R^n) is:

    Omega_R^n = \sqrt(delta ** 2 + 4ng_p^2)

"""


def Omega_R(Delta, hw_ph, g_ph, max_bosons=1):
    """The Rabi frequency
    Parameters:
        ----
        hw_ph: parameters
            energy of the cavit mode
        Delta: parameters
            gap between excited and ground state
        g_ph: float
            cavity molecule coupling strength
    """
    n = max_bosons
    delta = hw_ph - Delta
    return np.sqrt(delta**2 + 4 * n * g_ph**2)


def Emp(H_parameters, max_bosons=1):
    """
    The mixed energies
    Parameters
        ---
        H_parameters: tuple
            contains the central system model parameter
            H_parameters = Eg, Delta, hw_ph, g_ph
        max_bosons: integer
            number of photons in the cavity
    """
    n = max_bosons
    Eg, Delta, hw_ph, g_ph = H_parameters[:4]
    OmegaR = Omega_R(Delta, hw_ph, g_ph, n)
    Eplus = Eg + (Delta + (2 * n - 1) * hw_ph) / 2 + OmegaR / 2
    Eminus = Eg + (Delta + (2 * n - 1) * hw_ph) / 2 - OmegaR / 2
    return Eminus, Eplus


def theta(Delta, hw_ph, g_ph, max_bosons=1):
    """
    The angle mixed states
    Parameters:
        ----
        hw_ph: float
            energy of the cavit mode
        Delta: float
            gap between excited and ground state
        max_bosons: integer
            number of photons in the cavity
    """
    n = max_bosons
    delta = hw_ph - Delta

    if np.isclose(delta, 0):
        theta = np.pi / 4
    else:
        theta = 0.5 * np.arctan(2 * np.sqrt(n) * g_ph / delta)
    return theta


def check_degeneracy(arr):
    """Check if the array has any repeated elements (degeneracy)"""
    seen = set()
    for num in arr:
        if num in seen:
            return True  # Found a repeated element
        seen.add(num)
    return False  # No repeated elements


def E_index(H_parameters, Elist, states="bare"):
    """
    Arrangement of the energies
    Parameters:
        ----
        H_parameters: tuple
            contains the central system model parameter
            H_parameters = Eg, Delta, hw_ph, g_ph, U
        Elist: numpy array
            the energies of the system
        states: str
            'bare' means without interaction g_ph and 'dress' is with a finite interaction.
    """
    Eg, Delta, hw_ph, g_ph, U = H_parameters[:5]
    E0 = 0
    E1 = hw_ph
    Eg1 = Eg + hw_ph
    Ee = Eg + Delta
    Ee1 = Ee + hw_ph
    Ege = 2 * Eg + Delta + U
    Ege1 = Ege + hw_ph + U
    if states == "bare":
        E = np.array([E0, E1, Eg, Eg1, Ee, Ee1, Ege, Ege1])

        if check_degeneracy(E):
            return "Try non degenerate spectrum"

    if states == "dress":
        Eminus, Eplus = Emp(H_parameters)
        E = np.array([E0, E1, Eg, Eminus, Eplus, Ee1, Ege, Ege1])

        if check_degeneracy(E):
            return "Try non degenerate spectrum"
    L = []
    diff = np.abs(E[:, np.newaxis] - Elist)
    L = np.argmin(diff, axis=1)
    return L


def rho_arranged(H_parameters, VL, VR, kappa, Gamma_L, Gamma_R, kT, iva):
    """
    When we sweep one of the parameters in the Jaynes-Cummings model, we find that the arrangement of the energy levels changes, and therefore the labels of the density matrix also change. For this reason, we created this function, which ensures that even if the ordering of the energy levels changes, the labels of the density matrix remain consistent.
    Parameters:
        ----
        H_parameters: tuple
            contains the central system model parameters
            H_parameters = Eg, Delta, hw_ph, g_ph, U, max_bosons, True
        VL: float
            left chemical potential
        VR: float
            right chemical potential
        kappa: float
            damping rate
        Gamma_L: float
            left tunneling rates
        Gamma_R: float
            left tunneling rates
        kT: float
            Temperature
    """
    # We need the bare energies (without coupling)
    # H_parameters = Eg, Delta, hw_ph, g_ph, U, True
    # system hamiltonian
    H0, Hint, ops = tls.Hamiltonian(*H_parameters)

    if iva:
        H = H0
    else:
        H = H0 + Hint
    c_ops = tls.collapses(H, ops, VL, VR, kappa, Gamma_L, Gamma_R, kT)

    # density matrix
    L = no.liouvillian(H0 + Hint, list(c_ops))
    rho = nme.stationary(L)

    # list of bare energies
    if iva:
        E = H.toarray().diagonal()
        # we extract the index of the energies
        [i0, i1, ig, ig1, ie, ie1, ige, ige1] = E_index(H_parameters, E)
        P1 = rho[i1, i1]
        Pge1 = rho[ige1, ige1]
        Pg1 = rho[ig1, ig1]
        Pg1_e = rho[ig1, ie]
        Pe1 = rho[ie1, ie1]
        return P1, Pge1, Pg1, Pg1_e, Pe1

    else:
        E, _ = H.eigh()
        [i0, i1, ig, im, ip, ie1, ige, ige1] = E_index(H_parameters, E, "dress")
        P1 = rho[i1, i1]
        Pge1 = rho[ige1, ige1]
        Pm = rho[im, im]
        Pp = rho[ip, ip]
        Pe1 = rho[ie1, ie1]
    return P1, Pge1, Pm, Pp, Pe1


def spectrum_iva(
    H_parameters, VL, VR, kappa, Gamma_L, Gamma_R, kT, wlist, data=False, width=False
):
    """
    Spectrum analytical solution as sum of lorentzians
    Parameters
        ----
        H_parameters: tuple
            contains the central system model parameters
            H_parameters = Eg, Delta, hw_ph, g_ph, U
        VL: float
            left chemical potential
        VR: float
            right chemical potential
        kappa: float
            damping rate
        Gamma_L: float
            left tunneling rates
        Gamma_R: float
            left tunneling rates
        kT: float
            Temperature
        wlist: numpy array
            array with discretization of the energies
        data: logic
            the width and height position values of each Lorentzian that contributes to the spectrum.
        width: logic
            return the main width and energies
    """

    # H_parameters = Eg, Delta, hw_ph, g_ph, U, True
    H_parameters = H_parameters[:5]
    _, Delta, hw_ph, g_ph, _ = H_parameters
    H_parameters = *H_parameters, 1, True
    P1, Pge1, Pg1, Pg1_e, Pe1 = rho_arranged(
        H_parameters, VL, VR, kappa, Gamma_L, Gamma_R, kT, iva=True
    )

    OmegaR = Omega_R(Delta, hw_ph, g_ph)
    # this analytical solutions works for large bias limit VL>>VR
    # g0,g1 block
    # Eq 104, some necessary parameters for the eigenvalues
    a1 = 1j * hw_ph - (2 * Gamma_L + 2 * Gamma_R + kappa) / 2
    c1 = -1j * g_ph
    A1 = 1j * (hw_ph + Delta) - (4 * Gamma_L + 4 * Gamma_R + kappa) / 2
    B1 = np.sqrt(kappa**2 / 4 - OmegaR**2 - 1j * kappa * (hw_ph - Delta))

    # l1m and l1p are eigenvalues
    E1m = (A1 - B1) / 2
    E1p = (A1 + B1) / 2

    # left and right eigenvalues  Eqs 111--> 114
    v1m = [c1, E1m - a1]
    v1p = [c1, E1p - a1]
    v1 = np.stack([v1p, v1m]).T
    w1 = v1.conj()

    norm = np.einsum("ai,ai->i", w1.conj(), v1) ** 0.5
    v1 /= norm
    w1 /= norm.conj()

    # each row in v1 is a vector [[vp1x, v1mx], [v1py, v1my]]
    # coeficientes A and B eqs 117 and 122
    A1m = v1[0, 1]
    A1p = v1[0, 0]
    B1m = w1[0, 1].conjugate() * Pg1 + w1[1, 1].conjugate() * Pg1_e
    B1p = w1[0, 0].conjugate() * Pg1 + w1[1, 0].conjugate() * Pg1_e

    h1p = A1p * B1p
    h1m = A1m * B1m

    L1m = ndist.lorentzian(wlist - E1m.imag, -2 * E1m.real)
    L1p = ndist.lorentzian(wlist - E1p.imag, -2 * E1p.real)

    # eq 123
    peak1 = h1m * L1m + h1p * L1p

    # e0,e1 block
    a2 = 1j * hw_ph - (2 * Gamma_L + 2 * Gamma_R + kappa) / 2
    c2 = -1j * g_ph
    A2 = 1j * (hw_ph + Delta) - (4 * Gamma_L + 4 * Gamma_R + 3 * kappa) / 2
    B2 = np.sqrt(kappa**2 / 4 - OmegaR**2 - 1j * kappa * (hw_ph - Delta))

    E2p = (A2 + B2) / 2
    E2m = (A2 - B2) / 2

    # left and right eigenvalues  Eqs 111--> 114
    v2m = [c2, E2m - a2]
    v2p = [c2, E2p - a2]
    v2 = np.stack([v2p, v2m]).T
    w2 = v2.conj()

    norm = np.einsum("ai,ai->i", w2.conj(), v2) ** 0.5
    v2 /= norm
    w2 /= norm.conj()

    # coeficientes A and B eqs 117 and 122
    # each row in v2 is a vector [[vp2x, v2mx], [v2py, v2my]]
    A2m = v2[0, 1]
    A2p = v2[0, 0]
    B2m = w2[0, 1].conjugate() * Pe1
    B2p = w2[0, 0].conjugate() * Pe1

    h2p = A2p * B2p
    h2m = A2m * B2m

    L2p = ndist.lorentzian(wlist - E2p.imag, -2 * E2p.real)
    L2m = ndist.lorentzian(wlist - E2m.imag, -2 * E2m.real)

    peak2 = h2m * L2m + h2p * L2p

    # peak at hw_ph
    peak3 = (P1 + Pge1) * ndist.lorentzian(
        wlist - hw_ph, (2 * Gamma_L + 2 * Gamma_R + kappa)
    )
    A3 = 1
    width3 = -(2 * Gamma_L + 2 * Gamma_R + kappa) / 2

    h2p = A2p * B2p

    if data == True:
        print(f"1 {np.abs(A1m):.6f} {np.abs(B1m):.6f} {E1m.real:.6f} {E1m.imag:.6f}")
        print(f"2 {np.abs(A1p):.6f} {np.abs(B1p):.6f} {E1p.real:.6f} {E1p.imag:.6f}")
        print(f"3 {np.abs(A2m):.6f} {np.abs(B2m):.6f} {E2m.real:.6f} {E2m.imag:.6f}")
        print(f"4 {np.abs(A2p):.6f} {np.abs(B2p):.6f} {E2p.real:.6f} {E2p.imag:.6f}")
        print(f"5 {np.abs(A3):.6f} {np.abs(P1):.6f} {width3:.6f} {hw_ph:.6f}")
        print(f"6 {np.abs(A3):.6f} {np.abs(Pge1):.6f} {width3:.6f} {hw_ph:.6f}")
    I = kappa * np.real(peak1 + peak2 + peak3)
    if width:
        return I, [E1m.imag, E1p.imag, -2 * E1m.real, -2 * E1p.real]
    return I


def spectrum_sec(
    H_parameters, VL, VR, kappa, Gamma_L, Gamma_R, kT, wlist, data=False, width=False
):
    """
    Spectrum analytical solution as sum of lorentzians
    Parameters
        ----
        H_parameters: tuple
            contains the central system model parameters
            H_parameters = Eg, Delta, hw_ph, g_ph, U
        VL: float
            left chemical potential
        VR: float
            right chemical potential
        kappa: float
            damping rate
        Gamma_L: float
            left tunneling rates
        Gamma_R: float
            left tunneling rates
        kT: float
            Temperature
        wlist: numpy array
            array with discretization of the energies
        data: logic
            print the width and height position values of each Lorentzian that contributes to the spectrum.
        width: logic
            return the main width and energies
    """
    # H_parameters = Eg, Delta, hw_ph, g_ph, U, True
    H_parameters = H_parameters[:5]
    Eg, Delta, hw_ph, g_ph, _ = H_parameters
    H_parameters = *H_parameters, 1, True
    P1, Pge1, Pm, Pp, Pe1 = rho_arranged(
        H_parameters, VL, VR, kappa, Gamma_L, Gamma_R, kT, iva=False
    )

    OmegaR = Omega_R(Delta, hw_ph, g_ph)
    angle = theta(Delta, hw_ph, g_ph)
    Em, Ep = Emp((Eg, Delta, hw_ph, g_ph))
    # coeficientes A1, A2,  eqs 64, 65. 77, 78
    A1 = 1
    A2 = 1
    B1 = P1
    B2 = Pge1
    E1 = hw_ph
    E2 = hw_ph
    Gamma_1 = kappa + 4 * Gamma_L
    Gamma_2 = kappa + 4 * Gamma_R

    I = A1 * B1 * ndist.lorentzian(wlist - E1, Gamma_1)
    I += A2 * B2 * ndist.lorentzian(wlist - E2, Gamma_2)

    # coeficients A3, A4, B3, B4 eqs 66, 67, 88, 89
    A3 = np.sin(angle)
    A4 = -1j * np.sin(angle)
    B3 = np.sin(angle) * Pm
    B4 = 1j * np.sin(angle) * Pe1
    E3 = Em - Eg
    E4 = Em - Eg
    Gamma_3 = 2 * (Gamma_L + Gamma_R) + kappa * np.sin(angle) ** 2
    Gamma_4 = 2 * (Gamma_L + Gamma_R) + kappa * np.sin(angle) ** 2

    I += A3 * B3 * ndist.lorentzian(wlist - E3, Gamma_3)
    I += A4 * B4 * ndist.lorentzian(wlist - E4, Gamma_4)

    # coeficients A5, A6, B5, B6 eqs 68, 69, 97, 98
    A5 = np.cos(angle)
    A6 = 1j * np.cos(angle)
    B5 = np.cos(angle) * Pp
    B6 = -1j * np.cos(angle) * Pe1
    E5 = Ep - Eg
    E6 = Ep - Eg
    Gamma_5 = 2 * (Gamma_L + Gamma_R) + kappa * np.cos(angle) ** 2
    Gamma_6 = 2 * (Gamma_L + Gamma_R) + kappa * np.cos(angle) ** 2

    I += A6 * B5 * ndist.lorentzian(wlist - E5, Gamma_6)
    I += A6 * B6 * ndist.lorentzian(wlist - E6, Gamma_6)

    if data == True:
        print(f"1 {np.abs(A1):.6f} {np.abs(B1):.6f} {Gamma_1:.6f} {E1:.6f}")
        print(f"2 {np.abs(A2):.6f} {np.abs(B2):.6f} {Gamma_2:.6f} {E2:.6f}")
        print(f"1 {np.abs(A3):.6f} {np.abs(B3):.6f} {Gamma_3:.6f} {E3:.6f}")
        print(f"1 {np.abs(A4):.6f} {np.abs(B4):.6f} {Gamma_4:.6f} {E4:.6f}")
        print(f"1 {np.abs(A5):.6f} {np.abs(B5):.6f} {Gamma_5:.6f} {E5:.6f}")
        print(f"1 {np.abs(A6):.6f} {np.abs(B6):.6f} {Gamma_6:.6f} {E6:.6f}")

    if width:
        return kappa * I, [E3, E5, Gamma_3, Gamma_5]
    return kappa * I


def spectrum(
    H_parameters,
    VL,
    VR,
    kappa,
    Gamma_L,
    Gamma_R,
    kT,
    wlist,
    data=False,
    width=False,
    iva=False,
):
    if iva:
        return spectrum_iva(
            H_parameters, VL, VR, kappa, Gamma_L, Gamma_R, kT, wlist, data, width
        )
    else:
        return spectrum_sec(
            H_parameters, VL, VR, kappa, Gamma_L, Gamma_R, kT, wlist, data, width
        )
