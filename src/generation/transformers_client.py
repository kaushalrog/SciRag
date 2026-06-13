import logging
import time
import math
from typing import Generator, List, Dict, Optional
import numpy as np

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    torch = None

from src.generation.base_client import BaseLLMClient, GenerationResult

logger = logging.getLogger(__name__)

class TransformersClient(BaseLLMClient):
    """
    Local LLM Client using HuggingFace Transformers.
    Defaults to 4-bit quantization if bitsandbytes is available,
    otherwise loads in float16/bfloat16.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen1.5-0.5B-Chat",
        temperature: float = 0.1,
        max_tokens: int = 512,
        device: str = "auto",
        use_4bit: bool = False
    ) -> None:
        if torch is None:
            raise ImportError("Please install torch and transformers to use TransformersClient")
            
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        logger.info(f"Loading local model {model_id} (4bit={use_4bit})...")
        
        kwargs = {}
        if use_4bit:
            kwargs["load_in_4bit"] = True
            kwargs["device_map"] = "auto"
        else:
            kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            kwargs["device_map"] = device
            
