import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def make_bar_plot(metrics_means: dict, metrics_stds: dict, output_path: Path, output_name: str):
    plt.rcParams.update({
        'axes.labelsize': 14,     # Axis labels
        'axes.titlesize': 16,     # Title
        'xtick.labelsize': 12,    # X-axis tick labels
        'ytick.labelsize': 12,    # Y-axis tick labels
        'font.size': 12           # General text font size
    })        
    metrics = list(metrics_means.keys())
    mean_values = list(metrics_means.values())
    std_values = list(metrics_stds.values())

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    # Bar plot with error bars
    bars = ax.bar(x, mean_values, width, yerr=std_values, capsize=5, label='Mean')

    # Add labels, title, and custom x-axis tick labels
    ax.set_ylabel('Scores')
    ax.set_title('Metrics Mean with Standard Deviation')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=15, ha="right")
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    # Display the plot
    plt.tight_layout()
    plt.show()

    fig.savefig(output_path / output_name)
