import numpy as np

def zscore_standardize(X: list, axis: int = 0, eps: float = 1e-12) -> np.ndarray:
    x=np.array(X).astype(float)
    mean = np.mean(x, axis=axis, keepdims = 1)
    standard = np.std(x, axis=axis, keepdims = 1)
    z_score = np.where(standard > eps, (x - mean) / standard, 0.0)
    return z_score