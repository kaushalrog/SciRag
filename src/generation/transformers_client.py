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
            
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        self.model.eval()
        logger.info("Model loaded successfully.")

    def generate(
        self, 
        messages: List[Dict], 
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> GenerationResult:
        
        t0 = time.perf_counter()
        
        temp = temperature if temperature is not None else self.temperature
        max_new_tokens = max_tokens or self.max_tokens
        
        # Format messages into prompt
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_len = inputs.input_ids.shape[1]
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temp,
                do_sample=temp > 0,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        gen_sequences = outputs.sequences[:, input_len:]
        text = self.tokenizer.decode(gen_sequences[0], skip_special_tokens=True)
        
        # Calculate Entropy and Logprobs
        token_logprobs = []
        entropies = []
        sequence_score = 0.0
        
        if outputs.scores:
            for i, score_tensor in enumerate(outputs.scores):
                # score_tensor shape: (batch_size, vocab_size)
                logits = score_tensor[0].float()
                
                # Apply temperature
                if temp > 0:
                    logits = logits / temp
                    
                probs = torch.softmax(logits, dim=-1)
                log_probs = torch.log_softmax(logits, dim=-1)
                
                # Token entropy: -sum(p * log p). Avoid 0 * -inf = nan
                valid_mask = probs > 0
                entropy = -(probs[valid_mask] * log_probs[valid_mask]).sum().item()
                entropies.append(entropy)
                
                # Get the logprob of the token that was actually chosen
                token_id = gen_sequences[0][i].item()
                chosen_logprob = log_probs[token_id].item()
                sequence_score += chosen_logprob
                
                token_logprobs.append({
                    "token_id": token_id,
                    "token": self.tokenizer.decode([token_id]),
                    "logprob": chosen_logprob,
                    "entropy": entropy
                })
        
        mean_entropy = np.mean(entropies) if entropies else 0.0
        latency = (time.perf_counter() - t0) * 1000
        
        return GenerationResult(
            text=text,
            token_logprobs=token_logprobs,
            entropy=float(mean_entropy),
            sequence_score=float(sequence_score),
            latency_ms=latency,
            prompt_tokens=input_len,
            completion_tokens=len(gen_sequences[0])
        )

    def stream(
        self, 
        messages: List[Dict], 
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        # For simplicity, returning the full generation block in streaming format.
        # A real implementation would use TextIteratorStreamer.
        result = self.generate(messages, max_tokens=max_tokens, **kwargs)
        yield result.text
