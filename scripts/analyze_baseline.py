import sys
import json
from pathlib import Path
from tabulate import tabulate

def analyze_baseline():
    log_path = Path(__file__).parent.parent / "benchmark" / "failure_cases" / "failure_log.json"
    if not log_path.exists():
        print(f"No failure log found at {log_path}. Run the baseline study first.")
        return

    with open(log_path, "r", encoding="utf-8") as f:
        data = json.load(f)
