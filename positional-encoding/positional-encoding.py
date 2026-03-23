import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    # Write code here
    pos = np.arange(seq_len)[:, None]
    i = np.arange((d_model+1) // 2)[None, :]

    angle = pos / (base ** (2*i/d_model))
    
    pe = np.zeros((seq_len, d_model), dtype=float)
    pe[:, 0::2] = np.sin(angle)
    pe[:, 1::2] = np.cos(angle[:, :pe[:, 1::2].shape[1]])
    return pe