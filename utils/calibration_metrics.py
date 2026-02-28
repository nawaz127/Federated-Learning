import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

def compute_ece(y_true, y_pred, n_bins=15):
    # y_pred: predicted probabilities (shape: [num_samples, num_classes])
    # y_true: true labels (shape: [num_samples])
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    confidences = np.max(y_pred, axis=1)
    predictions = np.argmax(y_pred, axis=1)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_mask = (confidences > bins[i]) & (confidences <= bins[i+1])
        bin_size = np.sum(bin_mask)
        if bin_size > 0:
            bin_acc = np.mean(predictions[bin_mask] == y_true[bin_mask])
            bin_conf = np.mean(confidences[bin_mask])
            ece += (bin_size / len(y_true)) * np.abs(bin_acc - bin_conf)
    return ece

def compute_brier_score(y_true, y_pred):
    # y_pred: predicted probabilities for positive class (binary)
    # y_true: true binary labels
    return brier_score_loss(y_true, y_pred)

def compute_calibration_curve(y_true, y_pred, n_bins=15):
    # y_pred: predicted probabilities for positive class (binary)
    # y_true: true binary labels
    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=n_bins)
    return prob_true, prob_pred
