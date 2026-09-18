import numpy as np

def zscore_standardize(X: list, axis: int = 0, eps: float = 1e-12) -> np.ndarray:
    """
    Returns population Z-scores as a NumPy array matching the shape of X.
    """
    X=np.array(X).astype(float)
    mean = np.mean(X, axis=axis, keepdims=True)
    standard_deviation = np.std(X, axis=axis, keepdims=True)
    #np.where(condition, true_result, false_result)
    zscore = np.where(standard_deviation > eps, (X - mean) / standard_deviation, 0.0)
    return zscore