import numpy as np
from typing import List
from sklearn.metrics.pairwise import cosine_similarity

class ConfidenceEstimator:
    """
    Estimates confidence via a hybrid approach:
    1. Retrieval Confidence (RC): Average similarity of top-k retrieved chunks.
    2. Agreement Confidence (AC): Extent to which multi-source evidence agrees.
    3. Consistency Confidence (CC): Mean cosine similarity of multiple generated answers.
    """
    def __init__(self, weights: dict = None):
        self.weights = weights or {"rc": 0.5, "ac": 0.3, "cc": 0.2}
        
    def calculate_rc(self, retrieval_scores: List[float]) -> float:
        """Retrieval Confidence (RC)."""
        if not retrieval_scores:
            return 0.0
        return float(np.mean(retrieval_scores))
        
    def calculate_ac(self, source_agreements: List[float]) -> float:
        """
        Agreement Confidence (AC).
        source_agreements is a list of pairwise semantic similarities between evidence from different sources.
        """
        if not source_agreements:
            return 0.0
        return float(np.mean(source_agreements))
        
    def calculate_cc(self, answer_embeddings: np.ndarray) -> float:
        """
        Consistency Confidence (CC).
        answer_embeddings: shape (num_samples, embedding_dim)
        Returns the mean pairwise cosine similarity of the generated answers.
        """
        if answer_embeddings.shape[0] < 2:
            return 1.0 # Cannot measure variance with < 2 samples, assume consistent
            
        sim_matrix = cosine_similarity(answer_embeddings)
        # Extract upper triangle excluding diagonal
        upper_tri_indices = np.triu_indices_from(sim_matrix, k=1)
        mean_sim = np.mean(sim_matrix[upper_tri_indices])
        return float(mean_sim)
        
    def estimate_confidence(self, rc: float, ac: float, cc: float) -> float:
        """
        Combines the confidence components using the defined weights.
        """
        final_conf = (self.weights["rc"] * rc) + (self.weights["ac"] * ac) + (self.weights["cc"] * cc)
        return float(final_conf)
