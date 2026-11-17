import numpy as np

def prepend_class_token(patches: np.ndarray,
                        cls_token: np.ndarray) -> np.ndarray:
    """
    Returns the float64 sequence with the class token at position zero.
    """
    B = patches.shape[0] 
    D = patches.shape[2]
    
    # Broadcast the cls_token, NOT patches
    cls_expanded = np.broadcast_to(cls_token, (B, 1, D))
    
    # Glue them together along the sequence axis (axis=1)
    z = np.concatenate((cls_expanded, patches), axis=1)
    
    return z