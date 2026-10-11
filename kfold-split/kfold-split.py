import numpy as np

def kfold_split(N: int, k: int, shuffle: bool = True, seed: int = 0) -> list:
    """
    Returns a list of dictionaries with train_idx and val_idx.
    """
    array = np.arange(N)
    
    if shuffle:
        rng = np.random.default_rng(seed)
        array = rng.permutation(array)
        
    output = []
    q = N//k
    r = N%k
    start = 0


    for i in range(k) :
        if i < r :
            size = q + 1
        else:
            size = q
        end = start + size
        val = array[start: end]
        train = np.concatenate((array[:start], array[end:]))
        output.append({"train_idx": train, "val_idx": val})
        start = end  
        
    return output    