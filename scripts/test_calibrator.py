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
            levels[lvl] = {
                "correct": 0, 
                "base_confidences": [], 
                "calib_confidences": [], 
                "total": 0, 
                "base_fcr": 0,
                "calib_fcr": 0
            }
        
        is_correct = case["is_correct"]
        base_conf = case["confidence"]
        calib_conf = calibrator.predict_confidence(features[i])
        
        levels[lvl]["total"] += 1
        if is_correct:
            levels[lvl]["correct"] += 1
            
        levels[lvl]["base_confidences"].append(base_conf)
        levels[lvl]["calib_confidences"].append(calib_conf)
        
        if not is_correct and base_conf > 0.7:
            levels[lvl]["base_fcr"] += 1
            
        if not is_correct and calib_conf > 0.7:
            levels[lvl]["calib_fcr"] += 1

    # Print Comparison Table
    table_data = []
    headers = ["Level", "Accuracy", "Base Conf", "Calib Conf", "Base FCR", "Calib FCR"]

    for lvl in sorted(levels.keys()):
        stats = levels[lvl]
        accuracy = stats["correct"] / stats["total"]
        avg_base_conf = sum(stats["base_confidences"]) / len(stats["base_confidences"])
        avg_calib_conf = sum(stats["calib_confidences"]) / len(stats["calib_confidences"])
        
        base_fcr = stats["base_fcr"] / stats["total"]
        calib_fcr = stats["calib_fcr"] / stats["total"]
        
        table_data.append([
            lvl, 
            f"{accuracy*100:.1f}%", 
            f"{avg_base_conf:.4f}", 
            f"{avg_calib_conf:.4f}",
            f"{base_fcr*100:.1f}%",
            f"{calib_fcr*100:.1f}%"
        ])
        
    print("\n=== Calibrator vs Baseline ===")
    print(tabulate(table_data, headers=headers, tablefmt="github"))
    
    total_base_fcr = sum(s["base_fcr"] for s in levels.values()) / len(data)
    total_calib_fcr = sum(s["calib_fcr"] for s in levels.values()) / len(data)
    print(f"\nOverall Baseline FCR: {total_base_fcr*100:.1f}%")
    print(f"Overall Calibrated FCR: {total_calib_fcr*100:.1f}%")

if __name__ == "__main__":
    test_calibrator()
