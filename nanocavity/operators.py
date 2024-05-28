import numpy as np
import nanocavity.distributions as ndist
import secondquant as sq

#two level system coupled to single cavity mode
def H_tls(Eg, delta, omega, coupling, u=0, rwa=True, max_bosons=1, ret_nop=False):
    [dg, de, a], [Nfg, Nfe, Nb] = \
        sq.composite(fermion_modes=2, boson_modes=1, max_bosons=max_bosons)
    He = Eg * Nfg + (Eg +  delta) * Nfe + u * dg.d * de.d * de * dg
    Hp = omega * Nb
    H0 = He + Hp
    if rwa:
        Hint = coupling * (a.d * dg.d * de + a * de.d * dg)
    else:
        Hint = coupling * (a + a.d) * (dg.d * de + de.d * dg)
    H = H0 +  Hint
    L = [dg, de, a]
    if ret_nop:
        return H, L, [Nfg, Nfe, Nb]
    return H, L
