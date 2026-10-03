"""Reproducible experiment pipeline for learning-augmented disk scheduling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.features import FEATURE_NAMES, extract_features
from src.model import predict_scheduler, train_selector
from src.plots import generate_figures
from src.schedulers import best_scheduler, run_all_schedulers
from src.write_paper import write_term_paper
from src.workload import WorkloadSample, generate_workload, generate_workload_shift


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

SCHEDULER_NAMES = ["FCFS", "SSTF", "SCAN", "C-SCAN"]
WORKLOAD_TYPES = ["uniform", "localized", "directional", "bursty"]

PARAMETERS = {
    "min_cylinder": 0,
    "max_cylinder": 199,
    "initial_head": 50,
    "request_count": 40,
    "samples_per_workload_type": 160,
    "evaluation_windows_per_workload_type": 50,
    "workload_shift_windows_per_phase": 12,
    "training_seed_start": 1000,
    "evaluation_seed_start": 10000,
    "workload_shift_seed_start": 30000,
    "scan_direction": "up",
    "decision_tree_random_state": 307,
    "decision_tree_max_depth": 4,
    "decision_tree_min_samples_leaf": 8,
    "test_split": 0.25,
    "label_tie_breaking": "FCFS, then SSTF, then SCAN, then C-SCAN",
}


def _scheduler_rows(sample: WorkloadSample) -> dict[str, int]:
    results = run_all_schedulers(
        sample.requests,
        sample.initial_head,
        PARAMETERS["min_cylinder"],
        PARAMETERS["max_cylinder"],
        PARAMETERS["scan_direction"],
    )
    return {f"{name}_movement": results[name].total_movement for name in SCHEDULER_NAMES}


def _label_row(sample: WorkloadSample, sample_id: int) -> dict[str, object]:
    results = run_all_schedulers(
        sample.requests,
        sample.initial_head,
        PARAMETERS["min_cylinder"],
        PARAMETERS["max_cylinder"],
        PARAMETERS["scan_direction"],
    )
    features = extract_features(sample.requests, sample.initial_head)
    row: dict[str, object] = {
        "sample_id": sample_id,
        "workload_type": sample.workload_type,
        "seed": sample.seed,
        "initial_head": sample.initial_head,
        "request_count": len(sample.requests),
        "best_scheduler": best_scheduler(results),
    }
    row.update(features)
    row.update({f"{name}_movement": results[name].total_movement for name in SCHEDULER_NAMES})
    return row


def build_training_dataset() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    sample_id = 0
    for workload_index, workload_type in enumerate(WORKLOAD_TYPES):
        base_seed = PARAMETERS["training_seed_start"] + workload_index * 1000
        for offset in range(PARAMETERS["samples_per_workload_type"]):
            sample = generate_workload(
                workload_type,
                seed=base_seed + offset,
                count=PARAMETERS["request_count"],
                initial_head=PARAMETERS["initial_head"],
                min_cylinder=PARAMETERS["min_cylinder"],
                max_cylinder=PARAMETERS["max_cylinder"],
            )
            rows.append(_label_row(sample, sample_id))
            sample_id += 1
    return pd.DataFrame(rows)


def evaluate_selector(selector) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    window_id = 0
    for workload_index, workload_type in enumerate(WORKLOAD_TYPES):
        base_seed = PARAMETERS["evaluation_seed_start"] + workload_index * 1000
        for offset in range(PARAMETERS["evaluation_windows_per_workload_type"]):
            sample = generate_workload(
                workload_type,
                seed=base_seed + offset,
                count=PARAMETERS["request_count"],
                initial_head=PARAMETERS["initial_head"],
                min_cylinder=PARAMETERS["min_cylinder"],
                max_cylinder=PARAMETERS["max_cylinder"],
            )
            features = extract_features(sample.requests, sample.initial_head)
            results = run_all_schedulers(
                sample.requests,
                sample.initial_head,
                PARAMETERS["min_cylinder"],
                PARAMETERS["max_cylinder"],
                PARAMETERS["scan_direction"],
            )
            actual_best = best_scheduler(results)
            predicted = predict_scheduler(selector, features)
            row: dict[str, object] = {
                "window_id": window_id,
                "workload_type": sample.workload_type,
                "phase": "none",
                "seed": sample.seed,
                "predicted_scheduler": predicted,
                "actual_best_scheduler": actual_best,
                "prediction_correct": predicted == actual_best,
                "learned_selector_movement": results[predicted].total_movement,
                "oracle_best_movement": results[actual_best].total_movement,
            }
            row.update(features)
            row.update({f"{name}_movement": results[name].total_movement for name in SCHEDULER_NAMES})
            rows.append(row)
            window_id += 1
    return pd.DataFrame(rows)


def evaluate_workload_shift(selector) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    samples = generate_workload_shift(
        seed=PARAMETERS["workload_shift_seed_start"],
        windows_per_phase=PARAMETERS["workload_shift_windows_per_phase"],
        count=PARAMETERS["request_count"],
        initial_head=PARAMETERS["initial_head"],
        min_cylinder=PARAMETERS["min_cylinder"],
        max_cylinder=PARAMETERS["max_cylinder"],
    )
    for window_id, sample in enumerate(samples):
        features = extract_features(sample.requests, sample.initial_head)
        results = run_all_schedulers(
            sample.requests,
            sample.initial_head,
            PARAMETERS["min_cylinder"],
            PARAMETERS["max_cylinder"],
            PARAMETERS["scan_direction"],
        )
        actual_best = best_scheduler(results)
        predicted = predict_scheduler(selector, features)
        row: dict[str, object] = {
            "window_id": window_id,
            "phase": sample.phase,
            "workload_type": sample.workload_type,
            "seed": sample.seed,
            "predicted_scheduler": predicted,
            "actual_best_scheduler": actual_best,
            "prediction_correct": predicted == actual_best,
            "learned_selector_movement": results[predicted].total_movement,
            "oracle_best_movement": results[actual_best].total_movement,
        }
        row.update(features)
        row.update({f"{name}_movement": results[name].total_movement for name in SCHEDULER_NAMES})
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_results(raw_results: pd.DataFrame) -> pd.DataFrame:
    long_rows: list[dict[str, object]] = []
    for _, row in raw_results.iterrows():
        for scheduler in SCHEDULER_NAMES:
            long_rows.append(
                {
                    "workload_type": row["workload_type"],
                    "method": scheduler,
                    "total_movement": row[f"{scheduler}_movement"],
                }
            )
        long_rows.append(
            {
                "workload_type": row["workload_type"],
                "method": "Learned selector",
                "total_movement": row["learned_selector_movement"],
            }
        )
        long_rows.append(
            {
                "workload_type": row["workload_type"],
                "method": "Oracle best",
                "total_movement": row["oracle_best_movement"],
            }
        )

    long_results = pd.DataFrame(long_rows)
    summary = (
        long_results.groupby(["workload_type", "method"])["total_movement"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )
    summary.columns = [
        "workload_type",
        "method",
        "mean_total_movement",
        "std_total_movement",
        "min_total_movement",
        "max_total_movement",
        "window_count",
    ]
    return summary


def write_model_metrics(dataset: pd.DataFrame, trained) -> pd.DataFrame:
    rows: list[dict[str, object]] = [
        {"metric": "test_accuracy", "label": "", "value": trained.accuracy},
        {"metric": "train_size", "label": "", "value": trained.train_size},
        {"metric": "test_size", "label": "", "value": trained.test_size},
        {"metric": "tree_depth", "label": "", "value": trained.classifier.get_depth()},
        {"metric": "tree_leaves", "label": "", "value": trained.classifier.get_n_leaves()},
        {"metric": "max_depth", "label": "", "value": PARAMETERS["decision_tree_max_depth"]},
        {"metric": "min_samples_leaf", "label": "", "value": PARAMETERS["decision_tree_min_samples_leaf"]},
    ]
    label_counts = dataset["best_scheduler"].value_counts()
    for label in SCHEDULER_NAMES:
        rows.append({"metric": "label_count", "label": label, "value": int(label_counts.get(label, 0))})
    for feature, importance in zip(FEATURE_NAMES, trained.classifier.feature_importances_):
        rows.append({"metric": "feature_importance", "label": feature, "value": float(importance)})
    return pd.DataFrame(rows)


def write_confusion_matrix(trained) -> pd.DataFrame:
    return pd.DataFrame(trained.confusion, index=trained.confusion_labels, columns=trained.confusion_labels)


def run_pipeline() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    dataset = build_training_dataset()
    trained = train_selector(
        dataset,
        random_state=PARAMETERS["decision_tree_random_state"],
        max_depth=PARAMETERS["decision_tree_max_depth"],
        min_samples_leaf=PARAMETERS["decision_tree_min_samples_leaf"],
        test_size=PARAMETERS["test_split"],
    )
    raw_results = evaluate_selector(trained.classifier)
    summary = summarize_results(raw_results)
    shift_results = evaluate_workload_shift(trained.classifier)
    model_metrics = write_model_metrics(dataset, trained)
    confusion = write_confusion_matrix(trained)

    dataset.to_csv(RESULTS_DIR / "training_dataset.csv", index=False)
    raw_results.to_csv(RESULTS_DIR / "raw_results.csv", index=False)
    summary.to_csv(RESULTS_DIR / "summary_results.csv", index=False)
    model_metrics.to_csv(RESULTS_DIR / "model_metrics.csv", index=False)
    confusion.to_csv(RESULTS_DIR / "confusion_matrix.csv")
    shift_results.to_csv(RESULTS_DIR / "workload_shift_results.csv", index=False)
    (RESULTS_DIR / "experiment_parameters.json").write_text(json.dumps(PARAMETERS, indent=2), encoding="utf-8")

    generate_figures(RESULTS_DIR, FIGURES_DIR)
    write_term_paper()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the disk scheduling experiment pipeline.")
    parser.add_argument("--run", action="store_true", help="Run the full experiment pipeline.")
    args = parser.parse_args()
    if args.run:
        run_pipeline()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
