"""
End-to-End Benchmark Execution Pipeline with Isolated Experiment Runs.
Executes dataset loading, model training, anomaly generation, metric evaluation,
and figure rendering with unique timestamped/named runs to ensure full isolation.
"""

import os
import sys
import argparse
import time
import json
import shutil
from datetime import datetime
from typing import Dict, Any

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import pandas as pd
from tabulate import tabulate

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.registry import (
    get_dataset,
    get_model,
    get_metric,
    list_datasets,
    list_models,
    list_metrics
)
from src.utils.visualizer import (
    plot_pca_projections,
    plot_correlation_heatmaps,
    plot_downstream_utility,
    plot_training_losses,
    plot_feature_distributions
)


def parse_args():
    parser = argparse.ArgumentParser(description="End-to-End Industrial Tabular Generative Anomaly Benchmark")
    parser.add_argument("--dataset", type=str, default="skab",
                        help=f"Dataset plugin to load. Registered: {list_datasets()} (default: skab)")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to custom CSV dataset (automatically activates generic_csv plugin)")
    parser.add_argument("--target_col", type=str, default="anomaly",
                        help="Target anomaly column name for CSV (default: anomaly)")
    parser.add_argument("--normal_val", type=int, default=0,
                        help="Value representing normal non-defect sample (default: 0)")
    parser.add_argument("--models", nargs="+", default=["tvae", "fence_gan", "tab_ddpm"],
                        help=f"Models to benchmark. Registered: {list_models()}")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs per generative model (default: 20)")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size (default: 128)")
    parser.add_argument("--num_samples", type=int, default=2000,
                        help="Number of synthetic anomalies to generate per model (default: 2000)")
    parser.add_argument("--figure_dir", type=str, default="figure",
                        help="Base directory to save output figures (default: figure/)")
    parser.add_argument("--exp_dir", type=str, default="experiments",
                        help="Base directory to save experiment checkpoints and logs (default: experiments/)")
    parser.add_argument("--exp_name", type=str, default=None,
                        help="Custom experiment run name. If omitted, uses <dataset>_YYYYMMDD_HHMMSS")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    start_total_time = time.time()

    print("=" * 80)
    print("[*] INDUSTRIAL ANOMALY SYNTHESIS BENCHMARK PIPELINE")
    print("    Architecture: 'Everything is a Plugin' (Dataset | Model | Metric)")
    print("=" * 80)

    # 1. LOAD DATASET
    print(f"\n[1/5] Ingesting Dataset...")
    if args.csv:
        dataset_inst = get_dataset(
            "generic_csv",
            csv_path=args.csv,
            target_col=args.target_col,
            normal_val=args.normal_val,
            random_state=args.seed
        )
    else:
        dataset_inst = get_dataset(args.dataset)

    data_result = dataset_inst.load()

    # GENERATE UNIQUE ISOLATED RUN IDENTIFIER
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.exp_name.strip() if args.exp_name else f"{data_result.dataset_name}_{timestamp}"

    # Setup isolated directories for this specific experiment run
    run_exp_dir = os.path.join(args.exp_dir, "runs", run_name)
    run_ckpt_dir = os.path.join(run_exp_dir, "checkpoints")
    run_fig_dir = os.path.join(args.figure_dir, run_name)
    latest_fig_dir = os.path.join(args.figure_dir, "latest")

    os.makedirs(run_ckpt_dir, exist_ok=True)
    os.makedirs(run_fig_dir, exist_ok=True)
    os.makedirs(latest_fig_dir, exist_ok=True)

    print(f"  -> Experiment Run : {run_name}")
    print(f"  -> Run Artifacts  : {os.path.abspath(run_exp_dir)}")
    print(f"  -> Run Figures    : {os.path.abspath(run_fig_dir)}")
    print(f"  -> Dataset Name   : {data_result.dataset_name}")
    print(f"  -> Normal Train   : {data_result.X_train_norm.shape[0]} samples (100% OK, sensor features = {data_result.input_dim})")
    print(f"  -> Test Evaluation: {data_result.X_test.shape[0]} samples (Normal: {len(data_result.real_normals_test)}, Real Anomaly: {len(data_result.real_anomalies_test)})")
    print(f"  -> Sensor Features: {data_result.feature_names}")

    # 2. TRAIN MODELS & SYNTHESIZE ANOMALIES
    print(f"\n[2/5] Training Generative Models ({len(args.models)} models)...")
    synths_dict: Dict[str, np.ndarray] = {}
    losses_dict: Dict[str, Any] = {}
    fitted_models: Dict[str, Any] = {}

    for m_name in args.models:
        print(f"\n--- Training Plugin: {m_name.upper()} ---")
        start_t = time.time()
        model = get_model(m_name, input_dim=data_result.input_dim)
        fit_info = model.fit(
            data_result.X_train_norm,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        losses_dict[m_name] = fit_info or {}
        elapsed = time.time() - start_t
        print(f"  -> Trained in {elapsed:.2f}s")

        # Save checkpoint in run-isolated directory
        ckpt_path = os.path.join(run_ckpt_dir, f"{m_name}.pt")
        model.save(ckpt_path)
        print(f"  -> Checkpoint saved: {ckpt_path}")

        # Synthesize anomalies
        print(f"  -> Sampling {args.num_samples} synthetic anomalies...")
        synth_anom = model.sample(args.num_samples, X_ref=data_result.X_train_norm)
        synths_dict[m_name] = synth_anom
        fitted_models[m_name] = model

    # 3. METRIC EVALUATION
    print(f"\n[3/5] Evaluating Benchmark Metrics (Fidelity, Correlation, Utility)...")
    fidelity_metric = get_metric("fidelity")
    corr_metric = get_metric("correlation")
    utility_metric = get_metric("utility", random_state=args.seed)

    real_norm_test = data_result.real_normals_test
    real_anom_test = data_result.real_anomalies_test

    # Compute Baseline Isolation Forest utility (Unsupervised)
    from sklearn.ensemble import IsolationForest
    iso = IsolationForest(contamination=0.1, random_state=args.seed, n_jobs=-1).fit(data_result.X_train_norm)
    iso_scores = -iso.score_samples(data_result.X_test)
    from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
    iso_scaled = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min() + 1e-8)
    best_iso_f1 = max([f1_score(data_result.y_test, (iso_scaled >= th).astype(int), zero_division=0) for th in np.linspace(0.05, 0.95, 91)])

    utility_dfs = [
        pd.DataFrame([{
            "Model_Source": "Unsupervised Baseline",
            "Classifier": "IsolationForest",
            "F1_Default_0.5": float(f1_score(data_result.y_test, (iso.predict(data_result.X_test) == -1).astype(int), zero_division=0)),
            "Best_F1_Optimal": float(best_iso_f1),
            "Optimal_Threshold": 0.5,
            "ROC_AUC": float(roc_auc_score(data_result.y_test, iso_scores)),
            "PR_AUC": float(average_precision_score(data_result.y_test, iso_scores))
        }])
    ]

    # Supervised Oracle (Trained with real anomalies)
    oracle_res = utility_metric.evaluate(
        real_norm=data_result.X_train_norm,
        real_anom=real_anom_test,
        synth_anom=real_anom_test[:min(len(real_anom_test), args.num_samples)],
        feature_names=data_result.feature_names,
        X_test=data_result.X_test,
        y_test=data_result.y_test,
        model_name="Supervised Oracle (Real)"
    )
    utility_dfs.append(oracle_res["results_df"])

    summary_records = []
    synths_corr_dict: Dict[str, tuple[np.ndarray, float]] = {}

    for m_name, synth_data in synths_dict.items():
        # Fidelity
        fid_res = fidelity_metric.evaluate(
            real_norm=real_norm_test,
            real_anom=real_anom_test,
            synth_anom=synth_data,
            feature_names=data_result.feature_names
        )

        # Correlation
        corr_res = corr_metric.evaluate(
            real_norm=real_norm_test,
            real_anom=real_anom_test,
            synth_anom=synth_data,
            feature_names=data_result.feature_names
        )
        synths_corr_dict[m_name] = (corr_res["synth_pearson_corr"], corr_res["delta_pearson_frobenius"])

        # Utility
        ut_res = utility_metric.evaluate(
            real_norm=data_result.X_train_norm,
            real_anom=real_anom_test,
            synth_anom=synth_data,
            feature_names=data_result.feature_names,
            X_test=data_result.X_test,
            y_test=data_result.y_test,
            model_name=m_name
        )
        utility_dfs.append(ut_res["results_df"])

        summary_records.append({
            "Model": m_name,
            "Mean KS (↓)": round(fid_res["mean_ks"], 4),
            "Mean Wasserstein (↓)": round(fid_res["mean_wasserstein"], 4),
            "Δ Pearson Fro (↓)": round(corr_res["delta_pearson_frobenius"], 3),
            "Δ Spearman Fro (↓)": round(corr_res["delta_spearman_frobenius"], 3),
            "Best Downstream F1 (↑)": round(ut_res["best_f1"], 4),
            "Best Downstream AUC (↑)": round(ut_res["best_roc_auc"], 4)
        })

    # Combine utility results
    all_utility_df = pd.concat(utility_dfs, ignore_index=True)
    summary_df = pd.DataFrame(summary_records)

    # 4. DISPLAY & SAVE BENCHMARK TABLES
    print("\n[4/5] Benchmark Results Summary:")
    print(tabulate(summary_df, headers="keys", tablefmt="fancy_grid", showindex=False))

    # Save isolated run files
    summary_csv = os.path.join(run_exp_dir, "benchmark_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    utility_csv = os.path.join(run_exp_dir, "utility_results.csv")
    all_utility_df.to_csv(utility_csv, index=False)

    # Save run configuration and metadata
    run_meta = {
        "run_name": run_name,
        "timestamp": timestamp,
        "dataset": data_result.dataset_name,
        "features": data_result.feature_names,
        "models": args.models,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_samples": args.num_samples,
        "seed": args.seed,
        "total_duration_sec": round(time.time() - start_total_time, 2)
    }
    with open(os.path.join(run_exp_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    # Update global cross-run leaderboard
    leaderboard_csv = os.path.join(args.exp_dir, "runs_leaderboard.csv")
    leaderboard_rows = summary_df.copy()
    leaderboard_rows.insert(0, "Timestamp", timestamp)
    leaderboard_rows.insert(0, "Run_Name", run_name)
    leaderboard_rows.insert(2, "Dataset", data_result.dataset_name)

    if os.path.exists(leaderboard_csv):
        leaderboard_rows.to_csv(leaderboard_csv, mode="a", header=False, index=False)
    else:
        leaderboard_rows.to_csv(leaderboard_csv, mode="w", header=True, index=False)

    print(f"\n  -> Isolated Run Results: {summary_csv}")
    print(f"  -> Cross-run Leaderboard: {leaderboard_csv}")

    # 5. RENDER & SAVE FIGURES INTO figure/<run_name>/ and figure/latest/
    print(f"\n[5/5] Generating Visualizations into '{run_fig_dir}'...")
    real_pearson = np.nan_to_num(np.corrcoef(real_anom_test, rowvar=False))

    fig_paths = [
        plot_pca_projections(data_result.X_train_norm, real_anom_test, synths_dict, output_dir=run_fig_dir),
        plot_correlation_heatmaps(real_pearson, synths_corr_dict, data_result.feature_names, output_dir=run_fig_dir),
        plot_downstream_utility(all_utility_df, output_dir=run_fig_dir),
        plot_training_losses(losses_dict, output_dir=run_fig_dir),
        plot_feature_distributions(data_result.X_train_norm, real_anom_test, synths_dict, data_result.feature_names, output_dir=run_fig_dir)
    ]

    for p in fig_paths:
        if p and os.path.exists(p):
            print(f"  -> Generated: {p}")
            # Mirror to figure/latest/
            shutil.copy2(p, os.path.join(latest_fig_dir, os.path.basename(p)))

    print(f"  -> Mirrored to latest view: {os.path.abspath(latest_fig_dir)}")

    print("\n" + "=" * 80)
    print("[SUCCESS] PIPELINE COMPLETE! Experiment run saved independently.")
    print(f"   Run Directory    : {os.path.abspath(run_exp_dir)}")
    print(f"   Run Figures      : {os.path.abspath(run_fig_dir)}")
    print(f"   Latest Figures   : {os.path.abspath(latest_fig_dir)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
