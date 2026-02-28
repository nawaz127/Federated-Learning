import numpy as np

def aggregate_mean_std_across_runs(run_metrics):
    # run_metrics: list of dicts, each with keys like 'accuracy', 'macro_auc', etc.
    results = {}
    for key in run_metrics[0].keys():
        values = [run[key] for run in run_metrics]
        mean = np.mean(values)
        std = np.std(values)
        results[key] = {'mean': mean, 'std': std}
    return results
