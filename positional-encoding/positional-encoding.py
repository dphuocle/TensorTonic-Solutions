import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    # Write code here
    pos = np.arange(seq_len)[:, None]              # shape: (seq_len, 1)
    num_freqs = (d_model + 1) // 2
    i = np.arange(num_freqs)[None, :]              # shape: (1, num_freqs)

    denom = base ** (2 * i / d_model)              # shape: (1, num_freqs)
    angle = pos / denom                            # shape: (seq_len, num_freqs)

    pe = np.zeros((seq_len, d_model), dtype=float)
    pe[:, 0::2] = np.sin(angle)                    # các cột chẵn
    pe[:, 1::2] = np.cos(angle[:, :d_model // 2])  # các cột lẻ

    return pe