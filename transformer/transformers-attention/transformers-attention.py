import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    d_k = K.shape[-1]
    Kt = K.transpose (-2, -1)
    fraction = Q@Kt / math.sqrt(d_k)
    attention = F.softmax(fraction, dim=-1)@V
    return attention