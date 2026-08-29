import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    p_q=Q@W_q
    p_k=K@W_k
    p_v=V@W_v

    d_model = K.shape[-1]
    d_k = d_model // num_heads
    batch = p_q.shape[0]
    seq = p_q.shape[1]
    
    p_q_t = p_q.reshape(batch, seq, num_heads, d_k).transpose(0, 2, 1, 3)
    p_k_t = p_k.reshape(batch, seq, num_heads, d_k).transpose(0, 2, 1, 3)
    p_v_t = p_v.reshape(batch, seq, num_heads, d_k).transpose(0, 2, 1, 3)

    multi_scores=p_q_t@p_k_t.swapaxes(-1, -2)
    scaled_multi_scores = multi_scores / np.sqrt(d_k)
    multi_attention_weights = softmax(scaled_multi_scores)@p_v_t
    concat_output = multi_attention_weights.transpose(0, 2, 1, 3).reshape(batch, seq, d_model)

    return concat_output@W_o