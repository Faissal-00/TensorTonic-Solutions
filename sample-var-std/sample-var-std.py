import numpy as np

def sample_var_std(x: list) -> dict:
    """
    Returns a dictionary with variance and standard_deviation.
    """
    Dictio={} 
    variance=float(np.var(x, ddof=1)) 
    Std_deviation=float(np.std(x,ddof=1)) 
    Dictio['variance'] = variance
    Dictio['standard_deviation'] = Std_deviation
    return Dictio