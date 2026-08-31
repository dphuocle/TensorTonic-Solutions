import numpy as np

def cosine_similarity(a: list, b: list) -> float:
    """
    Returns the cosine similarity as a Python float.
    """
    # Write code here
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    cos_sim = float(np.dot(a, b) / (norm_a * norm_b))
    if norm_a==0 or norm_b==0:
        cos_sim = 0.0
    return cos_sim