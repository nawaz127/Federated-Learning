import matplotlib.pyplot as plt


def plot_xai_consistency(scores: list[float], save_path: str) -> None:
    if not scores:
        return
    plt.figure(figsize=(8, 6))
    plt.plot(scores, marker="o")
    plt.xlabel("Federated Round")
    plt.ylabel("CAM Similarity (SSIM)")
    plt.title("Cross-Round Explainability Stability")
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
