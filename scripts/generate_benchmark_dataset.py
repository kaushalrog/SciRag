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
        true_answer="5.0 million",
    )
    base_ev1 = Evidence("The 2025 official census reported the population of Metropolis at exactly 5.0 million.", "Census Bureau", 0.9)
    item1.evidence_by_level[ContradictionLevel.LEVEL_0_CLEAN] = [
        base_ev1,
        Evidence("Metropolis reached a milestone of 5.0 million residents this year.", "City Council", 0.8)
    ]
    item1.evidence_by_level[ContradictionLevel.LEVEL_1_NUMERIC] = [
        base_ev1,
        Evidence("Metropolis reached a milestone of 5.1 million residents this year.", "City Council", 0.8)
    ]
    dataset.add_item(item1)

    # Example 2: Entity Contradiction
    item2 = BenchmarkItem(
        id="q2",
        question="Who was appointed as the CEO of GlobalTech in January 2024?",
        true_answer="Alice Smith",
    )
    base_ev2 = Evidence("In January 2024, the board unanimously appointed Alice Smith as the new CEO of GlobalTech.", "Reuters", 0.95)
    item2.evidence_by_level[ContradictionLevel.LEVEL_0_CLEAN] = [
        base_ev2,
        Evidence("Alice Smith takes the helm as GlobalTech's CEO following the January board meeting.", "TechCrunch", 0.85)
    ]
    item2.evidence_by_level[ContradictionLevel.LEVEL_2_ENTITY] = [
        base_ev2,
        Evidence("Bob Jones takes the helm as GlobalTech's CEO following the January board meeting.", "TechCrunch", 0.85)
    ]
    dataset.add_item(item2)

    # Example 3: Semantic Contradiction
    item3 = BenchmarkItem(
        id="q3",
        question="What were the clinical trial results for the drug Novocillin?",
        true_answer="The drug was shown to be effective.",
    )
