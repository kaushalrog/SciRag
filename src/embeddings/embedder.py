from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

class DocumentEmbedder:
    """
    Embeds document chunks using a pre-trained model.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        
    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Returns embeddings as a numpy array.
        """
        return self.model.encode(texts)
