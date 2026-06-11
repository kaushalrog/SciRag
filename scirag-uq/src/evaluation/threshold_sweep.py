import numpy as np
import json
import matplotlib.pyplot as plt
from src.evaluation.metrics import abstention_metrics, hallucination_rate

def sweep_thresholds(y_true: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray):
    """
    Sweeps thresholds and calculates Abstention Precision, Recall, and remaining Hallucination Rate.
    """
    results = []
    
    for tau in thresholds:
        precision, recall = abstention_metrics(y_true, y_prob, tau)
        
        # Calculate hallucination rate on the answers the model chose NOT to abstain on
        answered_mask = (y_prob >= tau)
        if np.sum(answered_mask) > 0:
            remaining_hallucination_rate = hallucination_rate(y_true[answered_mask])
        else:
            remaining_hallucination_rate = 0.0 # No hallucinations if it abstains on everything
            
        coverage = np.mean(answered_mask)
        
        results.append({
            "tau": float(tau),
            "precision": precision,
            "recall": recall,
            "remaining_hallucination_rate": remaining_hallucination_rate,
            "coverage": coverage
        })
        
    return results

def plot_sweep_results(results: list, output_path: str = "results/threshold_sweep.png"):
    taus = [r['tau'] for r in results]
    precisions = [r['precision'] for r in results]
    recalls = [r['recall'] for r in results]
    hallucinations = [r['remaining_hallucination_rate'] for r in results]
    coverages = [r['coverage'] for r in results]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:red'
    ax1.set_xlabel('Abstention Threshold (tau)')
    ax1.set_ylabel('Rate', color=color)
    ax1.plot(taus, hallucinations, label='Hallucination Rate', color='red', linestyle='-')
    ax1.plot(taus, coverages, label='Coverage', color='orange', linestyle='--')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Precision / Recall', color=color)
    ax2.plot(taus, precisions, label='Abstention Precision', color='blue', linestyle='-')
    ax2.plot(taus, recalls, label='Abstention Recall', color='cyan', linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color)
    
    fig.tight_layout()
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=4)
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Saved threshold sweep plot to {output_path}")

if __name__ == "__main__":
    # Mock data for testing
    np.random.seed(42)
    n_samples = 100
    y_true_mock = np.random.randint(0, 2, n_samples)
    y_prob_mock = np.random.uniform(0, 1, n_samples)
    
    # Give the model some predictive power
    y_prob_mock[y_true_mock == 1] = np.clip(y_prob_mock[y_true_mock == 1] + 0.3, 0, 1)
    y_prob_mock[y_true_mock == 0] = np.clip(y_prob_mock[y_true_mock == 0] - 0.3, 0, 1)
    
    thresholds_mock = np.linspace(0.1, 0.9, 9)
    res = sweep_thresholds(y_true_mock, y_prob_mock, thresholds_mock)
    
    import os
    os.makedirs("results", exist_ok=True)
    plot_sweep_results(res)
    
    with open("results/threshold_sweep.json", "w") as f:
        json.dump(res, f, indent=2)
