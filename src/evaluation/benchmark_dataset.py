import json
import logging
from dataclasses import dataclass, field, asdict
from enum import IntEnum
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class ContradictionLevel(IntEnum):
    LEVEL_0_CLEAN = 0      # No contradiction
    LEVEL_1_NUMERIC = 1    # Differing exact numbers (e.g. 5M vs 5.1M)
    LEVEL_2_ENTITY = 2     # Differing subjects/entities (e.g. Alice vs Bob)
    LEVEL_3_SEMANTIC = 3   # Opposing semantic claims (e.g. effective vs ineffective)
    LEVEL_4_FACTUAL = 4    # Major factual conflict (e.g. mission succeeded vs failed)

@dataclass
class Evidence:
    text: str
    source: str = "Unknown"
    source_reliability: float = 1.0

@dataclass
class BenchmarkItem:
    id: str
    question: str
    true_answer: str
    # Maps a ContradictionLevel to the specific set of retrieved documents
    # to be used for the evaluation at that level.
    evidence_by_level: Dict[ContradictionLevel, List[Evidence]] = field(default_factory=dict)

    def get_evidence(self, level: ContradictionLevel) -> List[Evidence]:
        return self.evidence_by_level.get(level, [])

class BenchmarkDataset:
    """
    Manages the Contradiction Robustness Benchmark dataset.
    Items can be loaded from or saved to disk.
    """
    def __init__(self, name: str = "scirag_cdc_benchmark"):
        self.name = name
        self.items: List[BenchmarkItem] = []

    def add_item(self, item: BenchmarkItem):
        self.items.append(item)

    def save_to_disk(self, output_dir: str):
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        
        # We can save it as a single JSON file or multiple files depending on the scale.
        # Here we save it as a unified JSON to keep all levels of a question together.
        data = []
        for item in self.items:
            item_dict = {
                "id": item.id,
                "question": item.question,
                "true_answer": item.true_answer,
                "evidence_by_level": {
                    int(level): [asdict(e) for e in ev_list]
                    for level, ev_list in item.evidence_by_level.items()
                }
            }
            data.append(item_dict)

        file_path = path / f"{self.name}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved {len(self.items)} benchmark items to {file_path}")

    @classmethod
    def load_from_disk(cls, file_path: str) -> "BenchmarkDataset":
