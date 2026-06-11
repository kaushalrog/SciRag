import numpy as np
from typing import List, Tuple
from sklearn.metrics import roc_auc_score

def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Computes the Expected Calibration Error (ECE).
    """
    bins = np.linspace(0., 1., n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    
    ece = 0.0
    total_samples = len(y_prob)
    
    for i in range(n_bins):
        bin_idx = binids == i
        if np.sum(bin_idx) > 0:
            bin_acc = np.mean(y_true[bin_idx])
            bin_conf = np.mean(y_prob[bin_idx])
            bin_weight = np.sum(bin_idx) / total_samples
            ece += bin_weight * np.abs(bin_acc - bin_conf)
            
    return ece

def calculate_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Computes the Area Under the Receiver Operating Characteristic curve.
    """
    if len(np.unique(y_true)) < 2:
        return 0.5 # Default if all labels are the same
    return roc_auc_score(y_true, y_prob)

def abstention_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Tuple[float, float]:
    """
    Computes Abstention Precision and Abstention Recall.
    y_true: 1 if the answer is correct (non-hallucinated), 0 if it is a hallucination.
    y_prob: Confidence score.
    Abstention occurs when y_prob < threshold.
    """
    abstain = (y_prob < threshold).astype(int)
    hallucination = (y_true == 0).astype(int)
    
    true_positives = np.sum(abstain * hallucination) # Rightfully abstained on a hallucination
    predicted_positives = np.sum(abstain) # Total times the model abstained
    actual_positives = np.sum(hallucination) # Total hallucinations
    
    precision = true_positives / predicted_positives if predicted_positives > 0 else 0.0
    recall = true_positives / actual_positives if actual_positives > 0 else 0.0
    
    return precision, recall

def hallucination_rate(y_true: np.ndarray) -> float:
    """
    y_true: 1 if the answer is correct, 0 if it is a hallucination.
    """
    return 1.0 - np.mean(y_true)

def faithfulness(y_true: np.ndarray) -> float:
    """
    y_true: 1 if the answer is supported by evidence, 0 otherwise.
    """
    return np.mean(y_true)
