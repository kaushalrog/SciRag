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
    labels = []
    for case in data:
        # We stored confidence = 1 / (1 + entropy)
        # So entropy = (1 / confidence) - 1
        entropy = (1.0 / case["confidence"]) - 1.0
        contradiction_level = case["contradiction_level"]
        contradiction_count = 1 if contradiction_level > 0 else 0
        
        features.append([entropy, contradiction_level, contradiction_count])
        labels.append(1.0 if case["is_correct"] else 0.0)

    calibrator = SimpleCalibrator()
    calibrator.fit(features, labels)

    # Re-evaluate all cases
    levels = {}
    for i, case in enumerate(data):
        lvl = case["contradiction_level"]
        if lvl not in levels:
