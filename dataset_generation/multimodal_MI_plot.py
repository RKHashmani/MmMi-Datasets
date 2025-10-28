import numpy as np
import matplotlib.pyplot as plt
import os
import re

def extract_ix_theta(MI_results, MI_labels):
    """
    Extracts I(X_i; θ) values from MI_results and MI_labels.
    :param MI_results: Array of mutual information results.
    :param MI_labels: Array of mutual information labels.
    :return: Tuple of (ix_theta_indices, ix_theta_values)
    """
    ix_theta_indices = []
    ix_theta_values = []
    for idx, label in enumerate(MI_labels):
        if label.startswith("I(X_") and "; θ" in label:
            match = re.match(r"I\(X_(\d+); θ\)", label)
            if match:
                i = int(match.group(1))
                ix_theta_indices.append(i)
                ix_theta_values.append(MI_results[idx])
    sorted_indices = np.argsort(ix_theta_indices)
    ix_theta_indices = np.array(ix_theta_indices)[sorted_indices]
    ix_theta_values = np.array(ix_theta_values)[sorted_indices]
    return ix_theta_indices, ix_theta_values

def extract_ix1_xi(MI_results, MI_labels):
    """
    Extracts I(X_1; X_i) values from MI_results and MI_labels.
    :param MI_results: Array of mutual information results.
    :param MI_labels: Array of mutual information labels.
    :return: Tuple of (ix_indices, ix_values)
    """
    # This matches labels like I(X_1; X_3), I(X_1; X_4), ...
    ix_indices = []
    ix_values = []
    for idx, label in enumerate(MI_labels):
        # Match: I(X_1; X_i)
        match = re.match(r"I\(X_1; X_(\d+)\)", label)
        if match:
            i = int(match.group(1))
            ix_indices.append(i)
            ix_values.append(MI_results[idx])
    sorted_indices = np.argsort(ix_indices)
    ix_indices = np.array(ix_indices)[sorted_indices]
    ix_values = np.array(ix_values)[sorted_indices]
    return ix_indices, ix_values


def plot_overlay(
        npz_files,
        labels=None,
        extract_fn=None,
        ylabel="",
        title="",
        save_pdf=None,
        legend=True
):
    """
    Plots overlayed line plots from multiple .npz files containing mutual information results.

    :param npz_files: List of .npz file paths.
    :param labels: List of labels for each plot.
    :param extract_fn: Function to extract x and y values from MI_results and MI_labels.
    :param ylabel: String for the y-axis label.
    :param title: String for the plot title.
    :param save_pdf: Filename to save the plot as a PDF. If None, the plot is not saved.
    :param legend: Boolean indicating whether to display the legend.
    :return:
    """
    # Colors: Reds, Blues, Greens
    colors = [
        '#1a9850', '#66bd63', '#a6d96a',  # Greens (lightest is now a deeper lime)
        '#4575b4', '#91bfdb', '#74add1',  # Blues (lightest is now more saturated)
        '#d73027', '#fc8d59', '#fdae61',  # Reds (lightest is now deeper)
    ]

    # Line styles
    linestyles = ['solid', 'dashed', 'dotted']

    if legend:
        plt.figure(figsize=(7, 6))
        # plt.figure(figsize=(10, 6))
    else:
        plt.figure(figsize=(7, 4))
        # plt.figure(figsize=(10, 6))

    for idx, npz_file in enumerate(npz_files):
        data = np.load(npz_file, allow_pickle=True)
        MI_results = data["MI_results"]
        MI_labels = data["MI_labels"]
        x_indices, y_values = extract_fn(MI_results, MI_labels)
        label = labels[idx] if labels is not None else os.path.basename(os.path.dirname(npz_file))
        color = colors[idx % len(colors)]
        linestyle = linestyles[idx % 3]
        plt.plot(
            x_indices, y_values, marker='o', label=label,
            color=color, linestyle=linestyle, linewidth=2
        )
    plt.xlabel("i (Modality index)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    if legend:
        plt.legend(
            loc='upper center',
            bbox_to_anchor=(0.50, -0.15),  # (horizontal center, 15% below plot)
            ncol=3,
            # frameon=False,  # (optional) no legend box frame
        )

    plt.tight_layout()
    if save_pdf:
        plt.savefig(save_pdf, dpi=600)
    plt.show()

npz_files = [
    "../output_dir/datasets/datasets_4547693_within_2/multimodal_decay_DAG_alpha1.2_beta1.2/MMSweep_0.npz",
    "../output_dir/datasets/datasets_4547693_within_2/multimodal_decay_DAG_alpha1.2_beta1.4/MMSweep_0.npz",
    "../output_dir/datasets/datasets_4547693_within_2/multimodal_decay_DAG_alpha1.2_beta1.6/MMSweep_0.npz",
    "../output_dir/datasets/datasets_4547693_within_2/multimodal_decay_DAG_alpha1.4_beta1.2/MMSweep_0.npz",
    "../output_dir/datasets/datasets_4547693_within_2/multimodal_decay_DAG_alpha1.4_beta1.4/MMSweep_0.npz",
    "../output_dir/datasets/datasets_4547693_within_2/multimodal_decay_DAG_alpha1.4_beta1.6/MMSweep_0.npz",
    "../output_dir/datasets/datasets_4547693_within_2/multimodal_decay_DAG_alpha1.6_beta1.2/MMSweep_0.npz",
    "../output_dir/datasets/datasets_4547693_within_2/multimodal_decay_DAG_alpha1.6_beta1.4/MMSweep_0.npz",
    "../output_dir/datasets/datasets_4547693_within_2/multimodal_decay_DAG_alpha1.6_beta1.6/MMSweep_0.npz",
]

labels = [
    r"$\alpha=1.2,\ \beta=1.2$",
    r"$\alpha=1.2,\ \beta=1.4$",
    r"$\alpha=1.2,\ \beta=1.6$",
    r"$\alpha=1.4,\ \beta=1.2$",
    r"$\alpha=1.4,\ \beta=1.4$",
    r"$\alpha=1.4,\ \beta=1.6$",
    r"$\alpha=1.6,\ \beta=1.2$",
    r"$\alpha=1.6,\ \beta=1.4$",
    r"$\alpha=1.6,\ \beta=1.6$",
]

# Plot I(X_i; θ) vs. i

plt.rcParams.update({
    "font.size": 12,            # Default text size
    "axes.titlesize": 18,       # Title font size
    "axes.labelsize": 18,       # X/Y label font size
    "legend.fontsize": 12,      # Legend font size
    "xtick.labelsize": 14,      # X tick labels
    "ytick.labelsize": 14,      # Y tick labels
})

plot_overlay(
    npz_files,
    labels=labels,
    extract_fn=extract_ix_theta,
    ylabel=r"$I(\theta ; X_i)$",
    title=r"Mutual Information $I(\theta ; X_i)$ vs $i$",
    save_pdf="figs/theta_xi_plot.pdf",
    legend=True
)

# Plot I(X_1; X_i) vs. i
plot_overlay(
    npz_files,
    labels=labels,
    extract_fn=extract_ix1_xi,
    ylabel=r"$I(X_1; X_i)$",
    title=r"Mutual Information $I(X_1; X_i)$ vs $i$",
    save_pdf="figs/x1_xi_plot.pdf",
    legend=True
)
