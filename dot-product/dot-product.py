import numpy as np

def dot_product(x: list, y: list) -> float:
    """
    Returns the dot product as a float.
    """
    output = np.dot(np.array(x, dtype=float), np.array(y, dtype=float))
    # OR
    # output = np.array(x, dtype=float) @ np.array(y, dtype=float)
    return float(output)