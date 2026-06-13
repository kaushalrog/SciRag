import sys
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.generation.transformers_client import TransformersClient

logging.basicConfig(level=logging.INFO)
