import numpy as np

def patch_embed(image: np.ndarray, patch_size: int,
                W_proj: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """
    Returns float64 patch embeddings with shape (B, N, D).
    """
    P = patch_size
    Batch, H, W, C = image.shape
    
    # Crop to valid boundaries
    valid_H = (H // P) * P
    valid_W = (W // P) * P
    N = (valid_H // P) * (valid_W // P)
    image = image[:, :valid_H, :valid_W, :]
    
    xp = []    
    # Extract and flatten patches
    for b in range(Batch):
        for y in range(0, valid_H, P): 
            for x in range(0, valid_W, P): 
                square = image[b, y:y+P, x:x+P, :]
                xp.append(square.flatten())
                
    # Reshape array (do not transpose)
    xp = np.array(xp, dtype=np.float64).reshape(Batch, N, P*P*C)
    
    # Apply linear projection
    zp = xp @ W_proj + bias
    
    return zp