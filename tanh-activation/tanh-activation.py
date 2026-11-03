import numpy as np

def tanh(x: list) -> np.ndarray:
    """
    Returns a NumPy array with the same shape as x.
    """
    # X=np.array(x)
    # e_x=np.exp(X)
    # e__x=np.exp(-X)
    # tanhx=(e_x-e__x)/(e_x+e__x)
    # return tanhx
    
    return np.tanh(np.array(x, dtype=float))