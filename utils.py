def compute_accuracy(y_true, y_pred):
    return (y_true == y_pred).mean()