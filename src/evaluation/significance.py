import numpy as np
from scipy import stats
from typing import Tuple, Callable

def bootstrap_ci(data: np.ndarray, stat_func: Callable, alpha: float = 0.05, n_bootstraps: int = 1000) -> Tuple[float, float]:
    """
    Computes bootstrap confidence intervals for a given statistic.
    """
    bootstrapped_stats = []
    n = len(data)
    for _ in range(n_bootstraps):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrapped_stats.append(stat_func(sample))
        
    lower = np.percentile(bootstrapped_stats, 100 * (alpha / 2))
    upper = np.percentile(bootstrapped_stats, 100 * (1 - alpha / 2))
    return lower, upper

def paired_t_test(sample1: np.ndarray, sample2: np.ndarray) -> Tuple[float, float]:
    """
    Performs a paired t-test between two samples (e.g. error rates of Baseline vs SciRAG-UQ).
    Returns (t_statistic, p_value)
    """
    return stats.ttest_rel(sample1, sample2)

def wilcoxon_signed_rank_test(sample1: np.ndarray, sample2: np.ndarray) -> Tuple[float, float]:
    """
    Performs a Wilcoxon signed-rank test, useful when data is not normally distributed.
    Returns (statistic, p_value)
    """
    return stats.wilcoxon(sample1, sample2)

def test_significance(baseline_errors: np.ndarray, proposed_errors: np.ndarray, alpha: float = 0.05) -> dict:
    """
    Runs both paired t-test and Wilcoxon test to check for significant differences.
    """
    t_stat, p_ttest = paired_t_test(baseline_errors, proposed_errors)
    w_stat, p_wilcoxon = wilcoxon_signed_rank_test(baseline_errors, proposed_errors)
    
    return {
        "t_test_p_value": p_ttest,
        "wilcoxon_p_value": p_wilcoxon,
        "significant_ttest": p_ttest < alpha,
        "significant_wilcoxon": p_wilcoxon < alpha
    }
