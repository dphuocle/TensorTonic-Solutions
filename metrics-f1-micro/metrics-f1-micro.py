import numpy as np

def f1_micro(y_true, y_pred) -> float:
    """
    Compute micro-averaged F1 for multi-class integer labels.
    """
    # Write code here
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tp = np.count_nonzero(y_true == y_pred)
    fp = len(y_pred) - tp
    fn = len(y_true) - tp
    f1_micro = 2*tp / (2*tp + fp + fn)
    return f1_micro