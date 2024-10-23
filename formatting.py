import numpy as np

def scientific_format(num, r=1):
    # Special cases to avoid showing 1.0 in the scientific notation
    if num == 0:
        return r'$0$'
    
    # Convert the number to scientific notation
    formatted = '{:.1e}'.format(num)
    base, exponent = formatted.split('e')
    
    # Convert base and exponent to numbers
    base = float(base)
    exponent = int(exponent)

    if exponent==0:
        return round(num, r)
    
    # If the base is exactly 1.0, only use the exponent
    if base == 1.0:
        return r'$10^{{{}}}$'.format(exponent)
    else:
        # Otherwise, include the base and the exponent
        base = round(base, r)  # Round the base to one decimal place
        return r'${}\times 10^{{{}}}$'.format(base, exponent)

