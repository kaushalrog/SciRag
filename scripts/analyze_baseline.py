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

    # Group by level
    levels = {}
    for case in data:
        lvl = case["contradiction_level"]
        if lvl not in levels:
            levels[lvl] = {"correct": 0, "confidences": [], "total": 0, "fcr_count": 0}
        
        levels[lvl]["total"] += 1
        if case["is_correct"]:
            levels[lvl]["correct"] += 1
            
        levels[lvl]["confidences"].append(case["confidence"])
        
        if not case["is_correct"] and case["confidence"] > 0.7:
            levels[lvl]["fcr_count"] += 1

    # Print Table
    table_data = []
    headers = ["Level", "Accuracy", "Avg Confidence", "Confidence Gap", "FCR"]

    for lvl in sorted(levels.keys()):
        stats = levels[lvl]
        accuracy = stats["correct"] / stats["total"]
        avg_conf = sum(stats["confidences"]) / len(stats["confidences"])
        fcr = stats["fcr_count"] / stats["total"]
        gap = avg_conf - accuracy
        
        table_data.append([
            lvl, 
            f"{accuracy*100:.1f}%", 
            f"{avg_conf:.4f}", 
            f"{gap:.4f}",
            f"{fcr*100:.1f}%"
        ])
        
    print("\n=== Baseline Confidence vs Contradiction Level ===")
    print(tabulate(table_data, headers=headers, tablefmt="github"))
    
if __name__ == "__main__":
    analyze_baseline()
