import matplotlib.pyplot as plt
import numpy as np

def plot_iid_vs_noniid_histogram(client_distributions, alpha=None, save_path=None):
    # client_distributions: list of arrays (class proportions per client)
    all_props = np.concatenate(client_distributions)
    plt.figure()
    plt.hist(all_props, bins=20, alpha=0.7)
    plt.title('IID vs Non-IID Class Distribution')
    plt.xlabel('Class Proportion')
    plt.ylabel('Frequency')
    if alpha is not None:
        plt.suptitle(f'Dirichlet alpha={alpha}')
    if save_path:
        plt.savefig(save_path)
    plt.close()
