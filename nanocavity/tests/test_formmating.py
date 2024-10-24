import numpy as np
import nanocavity.formatting as ft


def test_formating():
    assert ft.scientific(1e-2) == "$10^{-2}$"  # avoid 1\times 10^-{2}
    assert ft.scientific(-1e-2) == "$-10^{-2}$"  # avoid -1\times 10^{-2}

    assert ft.scientific(4e-4) == "$4\\times 10^{-4}$"  # avoid 4.0
    assert ft.scientific(1.2e-4) == "$1.2\\times 10^{-4}$"

    assert ft.scientific(1.2) == "$1.2$"  # int
    assert ft.scientific(10) == "$10$"
    assert ft.scientific(1) == "$1$"

    assert ft.scientific(2.45434, 3) == "$2.454$"  # rounding

    array_numbers = np.array([1e-2, -1e-2, 4e-4, 0])

    # Array of numbers
    assert ft.scientific(array_numbers)  == ['$10^{-2}$', '$-10^{-2}$', '$4\\times 10^{-4}$', '$0$']

