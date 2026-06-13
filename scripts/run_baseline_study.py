import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.evaluation.benchmark_dataset import BenchmarkDataset, ContradictionLevel
from src.generation.transformers_client import TransformersClient
from src.evaluation.failure_cases import FailureCaseLogger
from src.evaluation.metrics import Evaluator, EvalSample

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_baseline_study():
    logger.info("Starting Baseline Confidence Study...")
    
    # 1. Load Dataset
    dataset_path = Path(__file__).parent.parent / "benchmark" / "scirag_cdc_benchmark.json"
    if not dataset_path.exists():
        logger.error("Benchmark dataset not found. Please run generate_benchmark_dataset.py first.")
        return
        
    dataset = BenchmarkDataset.load_from_disk(str(dataset_path))
    
    # 2. Initialize LLM
    llm = TransformersClient(model_id="Qwen/Qwen1.5-0.5B-Chat", use_4bit=False, device="cpu")
    
    # 3. Initialize Evaluator & Logger
    evaluator = Evaluator()
    failure_logger = FailureCaseLogger()
    
    samples_by_level = {level: [] for level in ContradictionLevel}

    # 4. Run Evaluation
    for item in dataset.items:
        for level in ContradictionLevel:
            evidence_list = item.get_evidence(level)
            if not evidence_list:
                continue # Skip if this level isn't defined for this item
                
            # Build Context
            context = "\n".join([f"- {ev.text}" for ev in evidence_list])
            
            # Form Prompt
            messages = [
                {"role": "system", "content": "You are a scientific assistant. Answer the user's question concisely based ONLY on the provided context."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {item.question}\nAnswer:"}
            ]
            
            # Generate
            logger.info(f"Generating answer for '{item.id}' at {level.name}...")
            result = llm.generate(messages, max_tokens=100)
