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
    
    # Extract and flatten patches (Manual sliding window)
    for b in range(Batch):
        for y in range(0, valid_H, P): 
            for x in range(0, valid_W, P): 
                square = image[b, y:y+P, x:x+P, :]
                xp.append(square.flatten())
                
    # Reshape array (do not transpose)
    xp = np.array(xp, dtype=np.float64).reshape(Batch, N, P*P*C)
    
    """
    # --- FAST VECTORIZED ALTERNATIVE (No for-loops) ---
    # grid_H, grid_W = valid_H // P, valid_W // P
    # xp = image.reshape(Batch, grid_H, P, grid_W, P, C)  # 1. 6D Split
    # xp = xp.transpose(0, 1, 3, 2, 4, 5)                 # 2. Axis Swap
    # xp = xp.reshape(Batch, N, P*P*C).astype(np.float64) # 3. Final Squash
    """
    
    # Apply linear projection
    zp = xp @ W_proj + bias
    
    return zp