import numpy as np
from typing import List, Dict

class EvidenceFusion:
    """
    Implements dynamic source weighting and evidence fusion.
    FusionScore_i = \alpha * RetrievalScore_i + \beta * ReliabilityScore_i + \gamma * CitationScore_i + \delta * RecencyScore_i
    w_i = FusionScore_i / \sum(FusionScore)
    """
    def __init__(self, alpha: float = 0.4, beta: float = 0.3, gamma: float = 0.2, delta: float = 0.1):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        
        # In a real scenario, this is learned from historical success rate.
        self.learned_reliability = {
            "arxiv": 0.85,
            "semantic_scholar": 0.92,
            "local_pdf": 0.65
        }
        
    def learn_reliability(self, source: str, success_count: int, total_count: int):
        """
        Updates the reliability score based on historical retrieval success rate.
        """
        if total_count > 0:
            self.learned_reliability[source] = success_count / total_count
            
    def compute_fusion_weights(self, retrieved_chunks: List[Dict]) -> np.ndarray:
        """
        Computes normalized fusion weights for each chunk.
        Expected keys in chunk dict: 'source', 'retrieval_score', 'citation_score', 'recency_score'
        """
        fusion_scores = []
        for chunk in retrieved_chunks:
            r_score = chunk.get('retrieval_score', 0.0)
            rel_score = self.learned_reliability.get(chunk.get('source', 'unknown'), 0.5)
            cit_score = chunk.get('citation_score', 0.0)
            rec_score = chunk.get('recency_score', 0.0)
            
            f_score = (self.alpha * r_score) + (self.beta * rel_score) + (self.gamma * cit_score) + (self.delta * rec_score)
            fusion_scores.append(f_score)
            
        fusion_scores = np.array(fusion_scores)
        total_score = np.sum(fusion_scores)
        if total_score > 0:
            weights = fusion_scores / total_score
        else:
            weights = np.ones_like(fusion_scores) / len(fusion_scores) if len(fusion_scores) > 0 else np.array([])
            
        return weights
