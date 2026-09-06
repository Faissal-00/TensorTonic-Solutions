import numpy as np

def matrix_inverse(A: list) -> np.ndarray | None:
    n = len(A)
    matrix = np.concatenate((A, np.eye(n)), axis=1).astype(float) 

    for i in range(n):
        current_column = matrix[i:, i]
        # Find the max absolute value and its correct row index
        max_val = np.abs(current_column).max()
        indice_max = np.argmax(np.abs(current_column))
        row_max = i + indice_max 
        # Check for singularity
        if max_val < 1e-12:
            return None
        # Swap the rows
        matrix[[i, row_max]] = matrix[[row_max, i]]
        ## Scale the pivot row to 1
        pivot = matrix[i, i]
        matrix[i] = matrix[i] / pivot
        # Eliminate other rows
        for j in range(n): 
            if j == i:
                continue
            multiplier = matrix[j][i] 
            matrix[j] = matrix[j] - (multiplier * matrix[i])
    inverse = matrix [:,n:] 
    return inverse