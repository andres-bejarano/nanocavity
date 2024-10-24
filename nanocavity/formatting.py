import numpy as np


def single_value(num, decimals=1, style="latex"):
    """
    Formats a single number in scientific notation in LaTeX or simple format.

    Parameters:
        -------------
        num: float
            Number to be formatted.
        decimals: int
            Number of decimals to keep.
        style: str
            Output format. Can be "latex" for LaTeX-style output, or "simple" for plain text.

    Returns:
        ----------
        str
        A string with the number in scientific notation in the chosen format.
    """
    # Special case for the number 0
    if num == 0:
        return r"$0$" if style == "latex" else "0"

    # Convert the number to scientific notation using absolute value to handle the sign separately
    formatted = "{:.1e}".format(abs(num))
    base, exponent = formatted.split("e")

    # Convert base and exponent to numbers
    base = float(base)
    exponent = int(exponent)

    # Handle the sign of the number
    sign = "-" if num < 0 else ""

    # If the exponent is 0, return the number without scientific notation
    if exponent == 0:
        if decimals == 0:
            return (
                f"${sign}{int(round(num, decimals))}$"
                if style == "latex"
                else f"{sign}{int(round(num, decimals))}"
            )
        else:
            return (
                f"${sign}{round(num, decimals)}$"
                if style == "latex"
                else f"{sign}{round(num, decimals)}"
            )

    if exponent == 1:
        return (
            f"${round(num, decimals)}$"
            if style == "latex"
            else f"{round(num, decimals)}"
        )

    # If the base is 1.0, only show the exponent (no base required)
    if base == 1.0:
        if style == "latex":
            return r"${}10^{{{}}}$".format(sign, exponent)
        else:
            return f"{sign}10^{exponent}"
    else:
        # Format the base and exponent depending on the chosen style
        base_str = (
            f"{int(base)}"
            if base.is_integer()
            else f"{base:.{decimals}f}".rstrip("0").rstrip(".")
        )

        if style == "latex":
            return r"${}{}\times 10^{{{}}}$".format(sign, base_str, exponent)
        else:
            return f"{sign}{base_str} \times  10^{exponent}"


def scientific(number, decimals=1, style="latex"):
    """
    Handles formatting of both single numbers and NumPy arrays in scientific notation in LaTeX or simple format.

    Parameters:
        -------------
        number: float or np.array
            Number or array to be converted.
        decimals: int
            Number of decimals to keep.
        style: str
            Output format. Can be "latex" for LaTeX-style output, or "simple" for plain text.

    Returns:
        ----------
        str or list of str
        Formatted string for a single number or array of such strings in the chosen format.
    """
    # If the input is a numpy array, apply the function element-wise and return as a Python list
    if isinstance(number, np.ndarray):
        # Ensure the output is a Python list, not a NumPy string array
        return [single_value(n, decimals, style) for n in number]

    # Otherwise, apply to a single number
    return single_value(number, decimals, style)
