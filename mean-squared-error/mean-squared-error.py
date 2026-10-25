import numpy as np

def mean_squared_error(y_pred: list, y_true: list) -> float:
    """
    Returns the error as a float.
    """
    y_pred = np.array(y_pred, dtype=float)
    y_true = np.array(y_true, dtype=float)
    N = len(y_pred)
    mse = ((y_pred-y_true)**2)/N
    return sum(mse)