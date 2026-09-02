import numpy as np

def apply_causal_mask(scores: list, mask_value: float = -1e9) -> np.ndarray:
    """
    Returns a causally masked NumPy array matching the shape of scores.
    """
    Matrix=np.array(scores)
    # dummy=np.ones_like(Matrix) #it creates matrix with same size of 1 s
    # map_grid = np.triu(dummy, k=1) #gets upper matrix and opposite is tril
    # Output = np.where(map_grid == 1, mask_value, Matrix)
    # return Output

    #OR memory-efficient way
    T = Matrix.shape[-1]  #find size (R, C) and -1 so it returns size of C
    mask = np.triu(np.ones((T, T), dtype=bool), k=1)
    Output = np.where(mask, mask_value, Matrix)
    return Output

    #!!!!!!!!!!! in triu or trill : 1 doesnt include diagonal, 0 yes, and -1 yes + other line