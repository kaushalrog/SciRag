import json
import random
from typing import List, Dict, Any

class DatasetBuilder:
    """
    Builds the 3-tier benchmark dataset for evaluating the Trustworthy Scientific QA framework.
    Tiers:
    1. Direct: Explicitly answerable from a single source.
    2. Synthesis: Requires fusing evidence from multiple sources.
    3. Unanswerable: Plausible-sounding questions that lack supporting evidence in the corpus.
    """

    def __init__(self, output_path: str = "data/benchmark.json"):
        self.output_path = output_path
        self.dataset: List[Dict[str, Any]] = []

    def add_direct_question(self, question: str, gold_answer: str, gold_evidence: List[str]):
        self.dataset.append({
            "question": question,
            "gold_answer": gold_answer,
            "gold_evidence": gold_evidence,
            "tier": "direct"
        })

    def add_synthesis_question(self, question: str, gold_answer: str, gold_evidence: List[str]):
        self.dataset.append({
            "question": question,
            "gold_answer": gold_answer,
            "gold_evidence": gold_evidence,
            "tier": "synthesis"
        })

    def add_unanswerable_question(self, question: str):
        self.dataset.append({
            "question": question,
            "gold_answer": "Insufficient supporting evidence found in the retrieved literature.",
            "gold_evidence": [],
            "tier": "unanswerable"
        })

    def build_mock_dataset(self):
        """Generates a mock dataset for initial pipeline testing."""
        # Direct
        self.add_direct_question(
            "What architecture does AntBot use for navigation?",
            "AntBot uses an optical compass architecture inspired by desert ants.",
            ["antbot_paper_2019_sec3"]
        )
        self.add_direct_question(
            "What is the maximum embedding length for all-MiniLM-L6-v2?",
            "The maximum sequence length is 256 tokens.",
            ["minilm_paper_sec2"]
        )
        
        # Synthesis
        self.add_synthesis_question(
            "Compare the retrieval latency of dense versus sparse methods as reported in recent literature.",
            "Dense methods typically show higher latency due to nearest neighbor search over high-dimensional vectors, whereas sparse methods like BM25 are faster due to inverted index lookups.",
            ["dense_retrieval_survey_2021", "bm25_optimization_2022"]
        )
        
        # Unanswerable (Plausible but unsupported)
        self.add_unanswerable_question(
            "What reinforcement learning reward function was used in AntBot?"
        )
        self.add_unanswerable_question(
            "Which quantum computing framework was evaluated for accelerating ChromaDB indexing?"
        )

        with open(self.output_path, 'w') as f:
            json.dump(self.dataset, f, indent=2)
        print(f"Generated {len(self.dataset)} benchmark questions to {self.output_path}")

if __name__ == "__main__":
    builder = DatasetBuilder(output_path="data/benchmark.json")
    builder.build_mock_dataset()
