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
            
            # Base Confidence from Entropy
            # Map entropy -> confidence (low entropy = high confidence)
            confidence = 1.0 / (1.0 + result.entropy)
            
            # Evaluate correct (simplified check)
            # A real implementation would use ROUGE-L or an LLM-as-a-judge
            is_correct = item.true_answer.lower() in result.text.lower() or Evaluator._rouge_l(result.text, item.true_answer) > 0.3
            
            # Log Failure Cases
            failure_logger.log_case(
                question=item.question,
                true_answer=item.true_answer,
                generated_answer=result.text,
                contradiction_level=int(level),
                confidence=confidence,
                is_correct=is_correct
            )
            
            sample = EvalSample(
                question=item.question,
                answer=result.text,
                context=context,
                reference=item.true_answer,
                confidence=confidence,
                abstained=False,
                truly_uncertain=False
            )
            samples_by_level[level].append((sample, 1.0 if is_correct else 0.0))

    # 5. Export Results & Summarize
    failure_logger.export()
    
    logger.info("=== Baseline Study Results ===")
    for level in ContradictionLevel:
        level_data = samples_by_level[level]
        if not level_data:
            continue
            
        samples = [s[0] for s in level_data]
        correctness = [s[1] for s in level_data]
        
        avg_conf = sum(s.confidence for s in samples) / len(samples)
        fcr = Evaluator._false_confidence_rate(samples, correctness, threshold=0.7)
        brier = Evaluator._brier_score(samples, correctness)
        
        logger.info(f"{level.name}:")
        logger.info(f"  Samples: {len(samples)}")
        logger.info(f"  Mean Confidence: {avg_conf:.4f}")
        logger.info(f"  False Confidence Rate (FCR): {fcr*100:.1f}%")
        logger.info(f"  Brier Score: {brier:.4f}")

if __name__ == "__main__":
    run_baseline_study()
