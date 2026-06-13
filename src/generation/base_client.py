from dataclasses import dataclass, field
from typing import Generator, List, Dict, Optional
from abc import ABC, abstractmethod

@dataclass
class GenerationResult:
    text: str
    token_logprobs: List[Dict] = field(default_factory=list)
    entropy: float = 0.0
    sequence_score: float = 0.0
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0

class BaseLLMClient(ABC):
    """
    Abstract base class for all LLM providers (Groq, Transformers, vLLM).
    Enforces a strict return type of `GenerationResult` which contains 
    critical UQ signals (entropy, token_logprobs, sequence_score).
    """

    @abstractmethod
    def generate(self, messages: List[Dict], max_tokens: Optional[int] = None, **kwargs) -> GenerationResult:
        """
        Synchronous generation.
