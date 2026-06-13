import sys
import json
from pathlib import Path
from tabulate import tabulate

sys.path.append(str(Path(__file__).parent.parent))
from src.uncertainty.calibrator import SimpleCalibrator

def test_calibrator():
    log_path = Path(__file__).parent.parent / "benchmark" / "failure_cases" / "failure_log.json"
    if not log_path.exists():
        print(f"No failure log found at {log_path}. Run the baseline study first.")
        return

    with open(log_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Prepare features and labels
    features = []
