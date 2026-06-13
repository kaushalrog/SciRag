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
