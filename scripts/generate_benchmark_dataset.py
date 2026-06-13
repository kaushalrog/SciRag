import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.evaluation.benchmark_dataset import BenchmarkDataset, BenchmarkItem, ContradictionLevel, Evidence

logging.basicConfig(level=logging.INFO)

def generate_samples():
    dataset = BenchmarkDataset("scirag_cdc_benchmark")

    # Example 1: Numeric Contradiction
    item1 = BenchmarkItem(
        id="q1",
        question="What is the population of Metropolis as of the 2025 census?",
