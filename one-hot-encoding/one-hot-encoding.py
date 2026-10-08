import numpy as np

def one_hot(y: list, num_classes=None) -> np.ndarray:
    """
    Returns a NumPy array with shape (N, K).
    """
    if num_classes is None :
        num_classes = max(y)+1
    matrix = np.zeros((len(y), num_classes), dtype=float) 

    rows = np.arange(len(y))

    matrix [[rows], [y]] = 1.0

    return matrix