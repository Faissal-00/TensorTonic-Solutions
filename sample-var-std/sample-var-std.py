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

    #ddof=0 (Population Variance): Divides the sum by exactly N. You use this when your dataset contains everything
    #ddof=1 (Sample Variance): Divides the sum by N-1. This is called Bessel's Correction. You use this when your data is just a sample of a larger population