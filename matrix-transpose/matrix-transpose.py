import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # 1. Convert the input into a NumPy array so we can read its size
    Matrix = np.array(A)
    Dim = Matrix.shape
    rows = Dim[0] 
    cols = Dim[1]
    # 3. Create a blank canvas filled with zeros. 
    # Notice we flip the shape here from (rows, cols) to (cols, rows)
    At = np.zeros((cols, rows)) 
    # 4. Look at every single row...
    for r in range(rows):
        # ...and look at every single column inside that row
        for c in range(cols):
            # 5. The core swap: 
            # Take the number at [row][col] and drop it into [col][row]
            At[c][r] = Matrix[r][c]
            
    # 6. Return the finished matrix AFTER all loops are done
    return At


"""
WHY WE USE TRANSPOSE:
1. Math: To align matrix shapes so PyTorch can multiply them without crashing.
2. Images: To flip image dimensions. PyTorch needs colors first (Channels, Height, Width) instead of last. Transpose fixes the order.
"""