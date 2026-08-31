import numpy as np

def dot_product(x: list, y: list) -> float:
    """
    Returns the dot product as a float.
    """
    # Write code here
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    dot = np.dot(x,y)
    dot = float(dot)
    return dot