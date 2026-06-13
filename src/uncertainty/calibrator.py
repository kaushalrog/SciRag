import logging
import numpy as np
from typing import List, Dict

try:
    from sklearn.linear_model import LogisticRegression
except ImportError:
    LogisticRegression = None

logger = logging.getLogger(__name__)

