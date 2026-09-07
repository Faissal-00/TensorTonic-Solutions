import numpy as np

def covariance_matrix(X: list) -> np.ndarray:
    """
    Returns the covariance matrix as a NumPy array.
    """
    N=len(X)
    matrix = np.array(X).astype(float)
    mean = np.mean(X, axis=0) 
    matrix_c= X-mean
    matrix_c_t=np.transpose(matrix_c)
    covariance = (matrix_c_t@matrix_c) / (N-1)
    return covariance