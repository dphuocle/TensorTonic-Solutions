import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    # Your code here
    if len(seqs) == 0:
        return np.zeros((0,0), dtype=int)

    N = len(seqs)
    if max_len is None :
        max_len = max(len(seq) for seq in seqs)
    pad = np.full((N, max_len), pad_value, dtype=int)

    for i, seq in enumerate(seqs):
        L = min(len(seq), max_len)
        pad[i, :L] = seq[:L]
    return pad