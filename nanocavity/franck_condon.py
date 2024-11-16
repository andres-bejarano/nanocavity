import numpy as np
from math import factorial


def FC_factor(n, m, g_ph):
    """Function to calculate the Franck-Condon Factors
    Parameters:
    --------------
        n: int
        bosonic quantum number
        m: int
        bosonic quantum number
        g_ph: float
        monopolar coupling
    Returns:
    --------------
        FC: float
        Franck-Condon Factor F_{nm}(g_ph)
    """
    if n > m:
        n, m = m, n
    FC_summand = (
        lambda k: (-(g_ph**2)) ** k
        * np.sqrt(factorial(n) * factorial(m))
        * g_ph ** (np.abs(n - m))
        * np.exp(-(g_ph**2) / 2)
        / factorial(k)
        / factorial(n - k)
        / factorial(k + np.abs(m - n))
    )
    FC = np.abs(sum(map(FC_summand, np.arange(n + 1)))) ** 2
    return FC
