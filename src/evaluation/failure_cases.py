import json
import logging
from dataclasses import dataclass, asdict
from typing import List
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class FailureCase:
    question: str
    true_answer: str
    generated_answer: str
    contradiction_level: int
    confidence: float
    is_correct: bool
    category: str

class FailureCaseLogger:
    """
    Utility to systematically log, categorize, and export qualitative failure cases.
    Tracks the crucial phenomenon: 'Contradictory evidence, still high confidence'.
    """
    def __init__(self, output_dir: str = "benchmark/failure_cases"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cases: List[FailureCase] = []

    def log_case(self, question: str, true_answer: str, generated_answer: str,
                 contradiction_level: int, confidence: float, is_correct: bool,
                 confidence_threshold: float = 0.7):
        
        # Categorize the prediction
        if is_correct and confidence >= confidence_threshold:
            category = "True Positive (Correct & Confident)"
        elif not is_correct and confidence < confidence_threshold:
            category = "True Negative (Wrong & Uncertain)"
        elif is_correct and confidence < confidence_threshold:
            category = "False Negative (Underconfident)"
        else: # not is_correct and confidence >= confidence_threshold
            if contradiction_level > 0:
                category = "Contradictory Evidence, Still High Confidence (Primary Failure)"
            else:
                category = "False Positive (Overconfident)"
        
        case = FailureCase(
            question=question,
            true_answer=true_answer,
            generated_answer=generated_answer,
            contradiction_level=contradiction_level,
            confidence=confidence,
            is_correct=is_correct,
            category=category
        )
        self.cases.append(case)
