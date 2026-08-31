import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    """
    Returns the transposed matrix as a NumPy array.
    """
    # Write code here
    A = np.array(A)
    n, m = A.shape
    T = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            T[i][j] = A[j][i]
    return T  