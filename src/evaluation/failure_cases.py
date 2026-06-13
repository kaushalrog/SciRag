import json
import logging
from dataclasses import dataclass, asdict
from typing import List
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class FailureCase:
    question: str
