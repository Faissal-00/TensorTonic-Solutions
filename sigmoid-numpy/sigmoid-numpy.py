import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """ 
    # Cast input to a NumPy array to handle scalars, lists, or matrices uniformly
    nbs = np.array(x)
    
    # Apply exponential and division element-wise across the entire tensor
    s = 1 / (1 + (np.exp(-nbs)))
    
    return s

"""
WHY WE USE SIGMOID:
It squashes raw network outputs to be exactly between 0 and 1. 
This lets you set a simple threshold (e.g., if output > 0.5, then it is a match).
"""
    