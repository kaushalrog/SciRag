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
