from collections import Counter
import numpy as np

def mean_median_mode(x: list) -> dict:
    """
    Returns a dictionary with mean, median, and mode.
    """
    Dictio={}
    Dictio['mean'] = float(np.mean(x)) 
    Dictio['median'] = float(np.median(x)) 

    vals, counts = np.unique(x,return_counts=True)
    mode_index = np.argmax(counts)
    Dictio['mode']=float(vals[mode_index]) 
    
    return Dictio