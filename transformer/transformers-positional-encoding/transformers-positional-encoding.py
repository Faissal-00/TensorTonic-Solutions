import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    # PE = np.zeros((seq_length, d_model))
    # for i in range (seq_length):
    #     for j in range(d_model):
    #         if j % 2 ==0 :
    #             pe = np.sin(i/(10000**(j/d_model)))
    #             PE[i][j] = pe
    #         elif j % 2 ==1 :
    #             pe = np.cos(i/(10000**((j-1)/d_model)))
    #             PE[i][j] = pe
    # return PE

    # Vectorization
    PE = np.zeros((seq_length, d_model))
    even_j = np.arange(0, d_model, 2)
    denominator = 10000 ** (even_j / d_model)
    positions = np.arange(seq_length)[:, np.newaxis]
    fraction = positions / denominator
    PE[:, 0::2] = np.sin(fraction)
    PE[:, 1::2] = np.cos(fraction)

    return PE