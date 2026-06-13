import sys
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.generation.transformers_client import TransformersClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def sanity_check():
    logger.info("Initializing model for sanity check...")
    client = TransformersClient(model_id="Qwen/Qwen1.5-0.5B-Chat", use_4bit=False, device="cpu")
    
    # Easy Prompt
    easy_msg = [{"role": "user", "content": "What is 2 + 2? Output just the number."}]
    res_easy = client.generate(easy_msg, max_tokens=10)
    
    # Hard Prompt
    hard_msg = [{"role": "user", "content": "Explain the exact mathematical derivation of quantum gravity in 5 sentences."}]
    res_hard = client.generate(hard_msg, max_tokens=50)
    
    print("\n--- SANITY CHECK RESULTS ---")
    print(f"EASY PROMPT: {res_easy.text.strip()}")
