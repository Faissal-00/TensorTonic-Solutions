import numpy as np

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Returns: Normalized array of same shape as x
    """
    d_model = x.shape[-1]
    variance = np.var(x, axis=-1, keepdims=True)
    mean = np.mean(x, axis=-1, keepdims=True)
    output = gamma*((x-mean)/np.sqrt(variance+eps))+beta
    return output