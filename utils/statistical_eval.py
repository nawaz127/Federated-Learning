import numpy as np
import scipy.stats as stats

def aggregate_runs(run_metrics):
    # run_metrics: list of dicts, each with keys like 'accuracy', 'macro_auc', etc.
    results = {}
    for key in run_metrics[0].keys():
        values = [run[key] for run in run_metrics]
        mean = np.mean(values)
        std = np.std(values)
        ci = stats.t.interval(0.95, len(values)-1, loc=mean, scale=stats.sem(values)) if len(values) > 1 else (mean, mean)
        results[key] = {'mean': mean, 'std': std, 'ci': ci}
    return results

def paired_t_test(run_metrics_a, run_metrics_b, key):
    # run_metrics_a/b: list of dicts
    values_a = [run[key] for run in run_metrics_a]
    values_b = [run[key] for run in run_metrics_b]
    t_stat, p_value = stats.ttest_rel(values_a, values_b)
    return t_stat, p_value
