import numpy as np

def apply_causal_mask(scores: list, mask_value: float = -1e9) -> np.ndarray:
    """
    Returns a causally masked NumPy array matching the shape of scores.
    """
    Matrix=np.array(scores)
    dummy=np.ones_like(Matrix) 
    map_grid = np.triu(dummy, k=1)
    Output = np.where(map_grid == 1, mask_value, Matrix)
    return Output