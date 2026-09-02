import numpy as np

def positional_encoding(seq_len: int, d_model: int, base: float = 10000.0) -> np.ndarray:
    """
    Returns a NumPy array of shape (seq_len, d_model).
    """
    pos=np.arange(seq_len).reshape(-1, 1)
    even=np.arange(0, d_model, 2)
    
    fraction = pos/(base**(even/d_model))
    
    PE = np.zeros((seq_len, d_model))
    PE[:,0::2]=np.sin(fraction) # [start:stop:step] all rows and columns 0, 2, 4....
    PE[:,1::2] = np.cos(fraction[:, :d_model//2]) # all rows and columns 1, 3, 5....
    
    return PE