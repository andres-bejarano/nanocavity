import numpy as np


def scientific_format(num, r=1):
    # Special case for the number 0
    if num == 0:
        return r"$0$"

    # Convert the number to scientific notation using absolute value to handle the sign separately
    formatted = "{:.1e}".format(abs(num))
    base, exponent = formatted.split("e")

    # Convert base and exponent to numbers
    base = float(base)
    exponent = int(exponent)

    # Handle the sign of the number
    sign = "-" if num < 0 else ""

    # If the exponent is 0, return the number as $X$ (where X is the rounded number)
    if exponent == 0:
        if r == 0:
            return f"${sign}{int(round(num, r))}$"  # Rounded to an integer, no decimal
        else:

            return f"${sign}{round(num, r)}$"  # Rounded to r
    # If the base is 1.0, only show the exponent (no base required)
    if base == 1.0:
        return r"${}10^{{{}}}$".format(sign, exponent)
    else:
        # If the base is an integer, show it without decimal places, otherwise round it to `r` places
        base_str = f"{int(base)}" if base.is_integer() else f"{base:.{r}f}"
        return r"${}{}\times 10^{{{}}}$".format(sign, base_str, exponent)
