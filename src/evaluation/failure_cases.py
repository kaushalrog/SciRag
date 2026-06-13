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
