import numpy as np

def softmax(x, axis=-1):
    """Provided: Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply layer normalization.
    """
    mean =np.mean(x, axis=-1, keepdims=True)
    variance =np.var(x, axis=-1, keepdims=True)
    output =gamma*((x-mean)/np.sqrt(variance+eps))+beta
    return output

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Multi-head attention.
    """
    d_model=Q.shape[-1]
    batch=Q.shape[0]
    seq_length=Q.shape[1]
    # or just this : batch, seq_length, d_model = Q.shape

    d_k=d_model//num_heads

    p_q=Q@W_q
    p_k=K@W_k
    p_v=V@W_v
    
    p_q_t= p_q.reshape(batch, seq_length, num_heads, d_k).transpose(0, 2, 1, 3)
    p_k_t= p_k.reshape(batch, seq_length, num_heads, d_k).transpose(0, 2, 1, 3)
    p_v_t= p_v.reshape(batch, seq_length, num_heads, d_k).transpose(0, 2, 1, 3)

    scores= p_q_t@p_k_t.swapaxes(-1, -2)
    scaled_scores = scores/np.sqrt(d_k)
    output=softmax(scaled_scores)@p_v_t
    concat=output.transpose(0, 2, 1, 3).reshape(batch, seq_length, d_model)
    
    return concat@W_o

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Position-wise feed-forward network.
    """
    hidden_layer=x@W1+b1
    output=np.maximum(0, hidden_layer)@W2+b2
    return output

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray,
                  b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Complete encoder block: MHA + FFN with residuals and layer norms.
    """
    mha_out = multi_head_attention(x, x, x, W_q, W_k, W_v, W_o, num_heads)
    x_prime = layer_norm(x + mha_out, gamma1, beta1)

    ff=feed_forward(x_prime, W1, b1, W2, b2)
    output=layer_norm(x_prime+ff, gamma2, beta2)
    return output