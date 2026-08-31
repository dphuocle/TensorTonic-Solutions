from collections import Counter
import numpy as np

def mean_median_mode(x: list) -> dict:
    """
    Returns a dictionary with mean, median, and mode.
    """
    # Write code here
    x = np.asarray(x, dtype=float)

    # Mean
    mean = sum(x) / len(x) # or mean = np.mean(x)

    # Median
    x_sorted = sorted(x)
    n = len(x_sorted)
    median = x_sorted[n // 2] if len(x_sorted) % 2 == 1 else (x_sorted[n // 2 - 1] + x_sorted[n // 2]) / 2
    # or median = np.median(x)

    # Mode
    counts = Counter(x)
    mode = counts.most_common(10)[0][0]
    
    return {
        "mean": float(mean),
        "median": float(median),
        "mode": float(mode)
    }