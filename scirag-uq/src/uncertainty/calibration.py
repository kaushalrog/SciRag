import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

class ConfidenceCalibrator:
    """
    Calibrates raw confidence scores using either Platt Scaling (Logistic Regression) 
    or Isotonic Regression, trained on the calibration dataset.
    """
    def __init__(self, method: str = 'isotonic'):
        self.method = method
        if method == 'platt':
            self.model = LogisticRegression(solver='lbfgs')
        elif method == 'isotonic':
            self.model = IsotonicRegression(out_of_bounds='clip')
        else:
            raise ValueError(f"Unknown calibration method: {method}")
            
        self.is_fitted = False
        
    def fit(self, raw_confidences: np.ndarray, labels: np.ndarray):
        """
        raw_confidences: 1D array of predicted raw confidences
        labels: 1D array of binary correctness labels (1 for correct, 0 for incorrect)
        """
        if self.method == 'platt':
            # LogisticRegression expects 2D array for X
            X = raw_confidences.reshape(-1, 1)
            self.model.fit(X, labels)
        else:
            self.model.fit(raw_confidences, labels)
        self.is_fitted = True
        
    def calibrate(self, raw_confidences: np.ndarray) -> np.ndarray:
        """
        Returns calibrated probabilities.
        """
        if not self.is_fitted:
            raise RuntimeError("Calibrator must be fitted before calling calibrate()")
            
        if self.method == 'platt':
            X = raw_confidences.reshape(-1, 1)
            return self.model.predict_proba(X)[:, 1]
        else:
            return self.model.predict(raw_confidences)
