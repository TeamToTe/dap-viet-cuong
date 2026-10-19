"""
Visualization utilities for Generative Industrial Anomaly Benchmark.
Saves publication-quality figures directly to the `figure/` directory.
"""

import os
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


def setup_plot_style():
    """Sets a clean, professional academic aesthetic."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.titlesize": 13,
        "figure.dpi": 300,
        "savefig.dpi": 300
    })


def plot_pca_projections(
    X_train_norm: np.ndarray,
    real_anom_test: np.ndarray,
    synths_dict: Dict[str, np.ndarray],
    output_dir: str = "figure"
) -> str:
    """
    Figure 1: 2D PCA Projections comparing Real Normal, Real Anomaly,
    and Synthetic Anomalies for each generative model.
    """
    os.makedirs(output_dir, exist_ok=True)
    setup_plot_style()

    pca = PCA(n_components=2, random_state=42).fit(X_train_norm)
    norm_2d = pca.transform(X_train_norm[:min(500, len(X_train_norm))])
    real_anom_2d = pca.transform(real_anom_test[:min(500, len(real_anom_test))])

    num_models = len(synths_dict)
    fig, axes = plt.subplots(1, max(1, num_models), figsize=(6 * max(1, num_models), 5), squeeze=False)
    axes = axes[0]

    colors = ["#2ca02c", "#ff7f0e", "#d62728", "#9467bd", "#8c564b"]

    for idx, (model_name, synth_data) in enumerate(synths_dict.items()):
        ax = axes[idx]
        synth_2d = pca.transform(synth_data[:min(500, len(synth_data))])
        color = colors[idx % len(colors)]

        ax.scatter(norm_2d[:, 0], norm_2d[:, 1], c="#1f77b4", alpha=0.3, s=20, label="Real Normal (OK)")
        ax.scatter(real_anom_2d[:, 0], real_anom_2d[:, 1], c="#7f7f7f", alpha=0.35, s=25, marker="x", label="Real Anomaly")
        ax.scatter(synth_2d[:, 0], synth_2d[:, 1], c=color, alpha=0.6, s=30, label=f"Synth: {model_name}")

        ax.set_title(f"PCA Projection: {model_name}", fontweight="bold")
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "01_pca_projection.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return save_path


def plot_correlation_heatmaps(
    real_corr: np.ndarray,
    synths_corr_dict: Dict[str, tuple[np.ndarray, float]],
    sensor_names: List[str],
    output_dir: str = "figure"
) -> str:
    """
    Figure 2: Heatmap grid comparing Pearson correlation matrices
    and reporting Frobenius distance error.
    """
    os.makedirs(output_dir, exist_ok=True)
    setup_plot_style()

    total_plots = 1 + len(synths_corr_dict)
    cols = 2
    rows = (total_plots + 1) // 2

    fig, axes = plt.subplots(rows, cols, figsize=(13, 5.5 * rows))
    axes = np.array(axes).reshape(-1)
    cmap = "coolwarm"

    # Plot Ground Truth
    sns.heatmap(
        real_corr, ax=axes[0], cmap=cmap, vmin=-1, vmax=1,
        xticklabels=sensor_names, yticklabels=sensor_names, cbar_kws={'shrink': 0.8}
    )
    axes[0].set_title("Ground Truth Real Anomaly Pearson Correlation", fontweight="bold")

    # Plot Model Synths
    for idx, (m_name, (synth_c, frob_val)) in enumerate(synths_corr_dict.items(), start=1):
        if idx < len(axes):
            sns.heatmap(
                synth_c, ax=axes[idx], cmap=cmap, vmin=-1, vmax=1,
                xticklabels=sensor_names, yticklabels=sensor_names, cbar_kws={'shrink': 0.8}
            )
            axes[idx].set_title(f"{m_name} (Δ Frobenius = {frob_val:.2f})", fontweight="bold")

    # Clean up empty subplots
    for i in range(total_plots, len(axes)):
        fig.delaxes(axes[i])

    for ax in axes[:total_plots]:
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "02_correlation_heatmaps.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return save_path


def plot_downstream_utility(
    utility_df: pd.DataFrame,
    output_dir: str = "figure"
) -> str:
    """
    Figure 3: Downstream Utility Comparison Bar Chart.
    F1-score across classifiers and synthetic models.
    """
    os.makedirs(output_dir, exist_ok=True)
    setup_plot_style()

    plt.figure(figsize=(11, 5.5))
    ax = sns.barplot(
        data=utility_df,
        x="Classifier",
        y="Best_F1_Optimal",
        hue="Model_Source",
        palette="Set2"
    )
    plt.title("Downstream Utility: Real Defect Detection F1-Score (Calibrated Threshold)", fontsize=12, fontweight="bold")
    plt.ylabel("Optimal F1-Score on Held-out Real Test Data")
    plt.ylim(0.0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)

    # Add numeric labels on top of bars
    for p in ax.patches:
        h = p.get_height()
        if not np.isnan(h) and h > 0:
            ax.annotate(f"{h:.2f}", (p.get_x() + p.get_width() / 2., h),
                        ha='center', va='bottom', fontsize=8, xytext=(0, 2),
                        textcoords='offset points')

    plt.tight_layout()
    save_path = os.path.join(output_dir, "03_downstream_utility.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return save_path


def plot_training_losses(
    losses_dict: Dict[str, Dict[str, Any]],
    output_dir: str = "figure"
) -> str:
    """
    Figure 4: Training Loss Curves for each generative model over epochs.
    """
    os.makedirs(output_dir, exist_ok=True)
    setup_plot_style()

    models = list(losses_dict.keys())
    n_models = len(models)
    if n_models == 0:
        return ""

    fig, axes = plt.subplots(1, n_models, figsize=(5.5 * n_models, 4.5), squeeze=False)
    axes = axes[0]

    for idx, (m_name, loss_info) in enumerate(losses_dict.items()):
        ax = axes[idx]
        if m_name == "fence_gan" and "loss_D" in loss_info and "loss_G" in loss_info:
            epochs_d = range(1, len(loss_info["loss_D"]) + 1)
            epochs_g = range(1, len(loss_info["loss_G"]) + 1)
            ax.plot(epochs_d, loss_info["loss_D"], label="Discriminator Loss", color="#1f77b4", lw=2)
            ax.plot(epochs_g, loss_info["loss_G"], label="Generator Fence Loss", color="#ff7f0e", lw=2)
            ax.set_ylabel("Loss")
        elif "loss" in loss_info:
            curve = loss_info["loss"]
            epochs_x = range(1, len(curve) + 1)
            color = "#2ca02c" if "vae" in m_name else "#d62728"
            ax.plot(epochs_x, curve, label=f"{m_name.upper()} Loss", color=color, lw=2)
            ax.set_ylabel("Training Loss")
        elif "loss_history" in loss_info:
            curve = loss_info["loss_history"]
            epochs_x = range(1, len(curve) + 1)
            ax.plot(epochs_x, curve, label=f"{m_name.upper()} Loss", color="#2ca02c", lw=2)
            ax.set_ylabel("Training Loss")
        elif "history" in loss_info and isinstance(loss_info["history"], list):
            curve = [h if isinstance(h, (int, float)) else h.get("loss", 0.0) for h in loss_info["history"]]
            epochs_x = range(1, len(curve) + 1)
            ax.plot(epochs_x, curve, label="Loss", lw=2)
            ax.set_ylabel("Loss")

        ax.set_title(f"Training Convergence: {m_name.upper()}", fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "04_training_losses.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return save_path


def plot_feature_distributions(
    X_train_norm: np.ndarray,
    real_anom_test: np.ndarray,
    synths_dict: Dict[str, np.ndarray],
    feature_names: List[str],
    output_dir: str = "figure",
    max_features: int = 8
) -> str:
    """
    Figure 5: Feature Marginal Distribution Curves (KDE).
    Compares Ground Truth Real Normal, Real Anomaly, and Synthesized Anomalies per feature.
    """
    os.makedirs(output_dir, exist_ok=True)
    setup_plot_style()

    n_feats = min(len(feature_names), max_features, X_train_norm.shape[1])
    cols = 4 if n_feats >= 4 else max(1, n_feats)
    rows = (n_feats + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.5 * rows), squeeze=False)
    axes_flat = axes.flatten()

    model_colors = {
        "tvae": "#2ca02c",
        "fence_gan": "#ff7f0e",
        "tab_ddpm": "#d62728"
    }
    fallback_colors = ["#9467bd", "#8c564b", "#e377c2", "#17becf"]

    for i in range(n_feats):
        ax = axes_flat[i]
        feat_name = feature_names[i]

        # Real Normal baseline
        sns.kdeplot(X_train_norm[:, i], ax=ax, label="Real Normal", color="#1f77b4", lw=1.8, linestyle=":")
        # Real Anomaly target
        if len(real_anom_test) > 0:
            sns.kdeplot(real_anom_test[:, i], ax=ax, label="Real Anomaly", color="#333333", lw=2.2, linestyle="-")

        # Synthetics
        for c_idx, (m_name, synth_data) in enumerate(synths_dict.items()):
            color = model_colors.get(m_name, fallback_colors[c_idx % len(fallback_colors)])
            sns.kdeplot(synth_data[:, i], ax=ax, label=f"Synth: {m_name}", color=color, lw=1.6, alpha=0.85)

        ax.set_title(f"{feat_name}", fontweight="bold", fontsize=10)
        ax.set_xlabel("Standardized Value", fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)
        if i == 0:
            ax.legend(loc="upper right", fontsize=7, framealpha=0.8)

    # Hide unused axes
    for j in range(n_feats, len(axes_flat)):
        fig.delaxes(axes_flat[j])

    plt.tight_layout()
    save_path = os.path.join(output_dir, "05_feature_distributions.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return save_path

