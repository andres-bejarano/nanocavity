import numpy as np

def single_value(num, decimals=1):
    """
    Formats a single number in scientific notation LaTeX format.
    
    Parameters:
        -------------
        num: float
            Number to be formatted
        decimals: int
            Number of decimals to keep
    Returns:
        ----------
        str
        A string with the number in scientific notation in LaTeX format.
    """
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
        if decimals == 0:
            return f"${sign}{int(round(num, decimals))}$"  # Rounded to an integer, no decimal
        else:
            return f"${sign}{round(num, decimals)}$"  # Rounded to specified decimals
    if exponent == 1:
        return f"${round(num, decimals)}$"
    # If the base is 1.0, only show the exponent (no base required)
    if base == 1.0:
        return r"${}10^{{{}}}$".format(sign, exponent)
    else:
        # If the base is an integer, show it without decimal places, otherwise round it to `decimals` places
        base_str = f"{int(base)}" if base.is_integer() else f"{base:.{decimals}f}".rstrip('0').rstrip('.')
        return r"${}{}\times 10^{{{}}}$".format(sign, base_str, exponent)


def scientific(number, decimals=1):
    """
    Handles formatting of both single numbers and NumPy arrays in scientific notation LaTeX format.
    
    Parameters:
        -------------
        number: float or np.array
            Number or array to be converted.
        decimals: int
            Number of decimals to keep
    Returns:
        ----------
        str or list of str
        LaTeX formatted string for a single number or array of such strings.
    """
    # If the input is a numpy array, apply the function element-wise and return as a Python list
    if isinstance(number, np.ndarray):
        # Ensure the output is a Python list, not a NumPy string array
        return [single_value(n, decimals) for n in number]

    # Otherwise, apply to a single number
    return single_value(number, decimals)
