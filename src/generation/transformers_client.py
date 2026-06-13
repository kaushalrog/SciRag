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
