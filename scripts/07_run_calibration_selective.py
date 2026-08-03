#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import GroupShuffleSplit, train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from health_misinfo.config import load_paths
from health_misinfo.evaluation.calibration import (
    calibration_metrics,
    choose_operating_thresholds,
    fit_calibrator,
    operating_point_result,
    reliability_table,
    selective_result,
    threshold_for_coverage,
)


def _plot_reliability(table: pd.DataFrame, path: Path) -> None:
    plotted = table.dropna(subset=["mean_probability", "observed_unreliable_rate"])
    fig, axis = plt.subplots(figsize=(5, 5))
    axis.plot([0, 1], [0, 1], linestyle="--", color="black", label="perfect calibration")
    axis.plot(
        plotted["mean_probability"],
        plotted["observed_unreliable_rate"],
        marker="o",
        label="selected calibration",
    )
    axis.set(xlabel="Mean predicted probability", ylabel="Observed unreliable rate", xlim=(0, 1), ylim=(0, 1))
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_risk_coverage(frame: pd.DataFrame, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(6, 4))
    axis.plot(frame["coverage"], frame["selective_risk"], marker="o")
    axis.set(xlabel="Coverage", ylabel="Selective risk", xlim=(0, 1))
    axis.invert_xaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _calibration_partitions(
    validation: pd.DataFrame,
    binary: np.ndarray,
    group_field: str,
) -> tuple[np.ndarray, np.ndarray, str]:
    groups = validation[group_field].fillna("").astype(str)
    if groups.nunique() >= 4:
        for attempt in range(100):
            splitter = GroupShuffleSplit(
                n_splits=1,
                test_size=0.5,
                random_state=613 + attempt,
            )
            fit_indices, selection_indices = next(
                splitter.split(validation, binary, groups=groups)
            )
            if (
                len(np.unique(binary[fit_indices])) == 2
                and len(np.unique(binary[selection_indices])) == 2
            ):
                return fit_indices, selection_indices, f"group_disjoint:{group_field}"
    all_indices = np.arange(len(validation))
    fit_indices, selection_indices = train_test_split(
        all_indices,
        test_size=0.5,
        random_state=613,
        stratify=binary,
    )
    return fit_indices, selection_indices, "record_stratified_fallback"


def _run_experiment(experiment_dir: Path) -> dict | None:
    config_path = experiment_dir / "config.yaml"
    validation_path = experiment_dir / "validation_predictions.csv"
    test_path = experiment_dir / "predictions.csv"
    if not (config_path.exists() and validation_path.exists() and test_path.exists()):
        return None
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("model") not in {"logistic_regression", "linear_svm"}:
        return None

    validation = pd.read_csv(validation_path)
    test = pd.read_csv(test_path)
    validation_binary = validation["harmonized_label"].eq("unreliable").astype(int).to_numpy()
    test_binary = test["harmonized_label"].eq("unreliable").astype(int).to_numpy()
    validation_scores = validation["score_unreliable"].to_numpy()
    test_scores = test["score_unreliable"].to_numpy()

    group_field = str(config["group_field"])
    calibration_fit_indices, selection_indices, partition_method = _calibration_partitions(
        validation,
        validation_binary,
        group_field,
    )
    fit_scores = validation_scores[calibration_fit_indices]
    fit_binary = validation_binary[calibration_fit_indices]
    selection_scores = validation_scores[selection_indices]
    selection_binary = validation_binary[selection_indices]

    methods = ["platt"]
    scores_are_probabilities = bool(
        np.all((validation_scores >= 0) & (validation_scores <= 1))
        and config.get("model") == "logistic_regression"
    )
    if scores_are_probabilities:
        methods.insert(0, "uncalibrated")
    class_counts = pd.Series(fit_binary).value_counts()
    if (
        len(calibration_fit_indices) >= 1_000
        and class_counts.get(0, 0) >= 200
        and class_counts.get(1, 0) >= 200
    ):
        methods.append("isotonic")

    candidates = []
    calibrators = {}
    for method in methods:
        calibrator = fit_calibrator(method, fit_scores, fit_binary)
        calibrators[method] = calibrator
        probabilities = calibrator.predict(selection_scores)
        candidates.append(
            {
                "method": method,
                "calibration_fit_records": len(calibration_fit_indices),
                "method_selection_records": len(selection_indices),
                "selection_split": "held_out_validation_selection_half",
                "partition_method": partition_method,
                **calibration_metrics(selection_binary, probabilities),
            }
        )
    candidate_frame = pd.DataFrame(candidates).sort_values(["brier_score", "method"])
    selected_method = str(candidate_frame.iloc[0]["method"])
    selected = calibrators[selected_method]
    test_probabilities = selected.predict(test_scores)
    selection_probabilities = selected.predict(selection_scores)

    out = experiment_dir / "calibration_selective"
    out.mkdir(parents=True, exist_ok=True)
    candidate_frame.to_csv(out / "validation_calibration_selection.csv", index=False)
    test_calibration = {
        "method": selected_method,
        "selection_rule": "lowest Brier score on held-out validation selection half",
        **calibration_metrics(test_binary, test_probabilities),
    }
    if scores_are_probabilities:
        test_calibration.update(
            {
                f"raw_{key}": value
                for key, value in calibration_metrics(test_binary, test_scores).items()
            }
        )
    pd.DataFrame([test_calibration]).to_csv(out / "test_calibration_metrics.csv", index=False)
    thresholds = choose_operating_thresholds(selection_binary, selection_probabilities)
    primary_decision_threshold = thresholds["macro_f1_validation_optimum"]
    calibrated_predictions = test.copy()
    calibrated_predictions["calibrated_probability_unreliable"] = test_probabilities
    calibrated_predictions["calibrated_prediction"] = np.where(
        test_probabilities >= primary_decision_threshold,
        "unreliable",
        "reliable",
    )
    calibrated_predictions["confidence"] = np.maximum(test_probabilities, 1 - test_probabilities)
    calibrated_predictions["calibrated_error"] = (
        calibrated_predictions["calibrated_prediction"]
        != calibrated_predictions["harmonized_label"]
    )
    calibrated_predictions.to_csv(out / "test_calibrated_predictions.csv", index=False)
    reliability = reliability_table(test_binary, test_probabilities)
    reliability.to_csv(out / "reliability_table.csv", index=False)
    _plot_reliability(reliability, out / "reliability_diagram.png")

    confidence = np.maximum(test_probabilities, 1 - test_probabilities)
    pd.DataFrame({"confidence": confidence}).to_csv(out / "confidence_values.csv", index=False)
    histogram, edges = np.histogram(confidence, bins=np.linspace(0.5, 1.0, 11))
    pd.DataFrame(
        {
            "lower": edges[:-1],
            "upper": edges[1:],
            "records": histogram,
        }
    ).to_csv(out / "confidence_histogram.csv", index=False)

    fixed_rows = []
    full_prediction = (test_probabilities >= primary_decision_threshold).astype(int)
    random_deferral_risk = float((full_prediction != test_binary).mean())
    for target_coverage in (1.0, 0.9, 0.8, 0.7):
        threshold = threshold_for_coverage(
            selection_probabilities,
            target_coverage,
        )
        fixed_rows.append(
            {
                "target_validation_coverage": target_coverage,
                "random_deferral_expected_risk": random_deferral_risk,
                **selective_result(
                    test_binary,
                    test_probabilities,
                    threshold,
                    decision_threshold=primary_decision_threshold,
                ),
            }
        )
    pd.DataFrame(fixed_rows).to_csv(out / "fixed_coverage_results.csv", index=False)

    curve_rows = []
    for target_coverage in np.linspace(1.0, 0.1, 19):
        threshold = threshold_for_coverage(selection_probabilities, float(target_coverage))
        curve_rows.append(
            {
                "target_validation_coverage": target_coverage,
                "random_deferral_expected_risk": random_deferral_risk,
                **selective_result(
                    test_binary,
                    test_probabilities,
                    threshold,
                    decision_threshold=primary_decision_threshold,
                ),
            }
        )
    curve = pd.DataFrame(curve_rows)
    curve.to_csv(out / "risk_coverage_curve.csv", index=False)
    _plot_risk_coverage(curve, out / "risk_coverage_curve.png")

    operating_rows = []
    for name, threshold in thresholds.items():
        operating_rows.append(
            {
                "operating_point": name,
                "threshold_selected_on": "validation",
                **operating_point_result(test_binary, test_probabilities, threshold),
            }
        )
    pd.DataFrame(operating_rows).to_csv(out / "operating_points.csv", index=False)
    coverage_threshold = threshold_for_coverage(selection_probabilities, 0.8)
    audit = calibrated_predictions.copy()
    audit["deferred_at_80pct_target"] = audit["confidence"] < coverage_threshold
    subgroup_rows = []
    for field in ("harmonized_label", "text_unit"):
        for value, subgroup in audit.groupby(field):
            subgroup_rows.append(
                {
                    "subgroup_field": field,
                    "subgroup_value": value,
                    "records": len(subgroup),
                    "deferred_records": int(subgroup["deferred_at_80pct_target"].sum()),
                    "deferral_rate": float(subgroup["deferred_at_80pct_target"].mean()),
                }
            )
    pd.DataFrame(subgroup_rows).to_csv(out / "deferral_subgroup_audit.csv", index=False)
    (out / "selected_calibration.json").write_text(
        json.dumps(
            {
                "selected_method": selected_method,
                "selection_rule": "lowest Brier score on held-out validation selection half",
                "calibration_fit_records": len(calibration_fit_indices),
                "method_selection_records": len(selection_indices),
                "partition_method": partition_method,
                "primary_decision_threshold": primary_decision_threshold,
                "isotonic_eligible": "isotonic" in methods,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "experiment_id": config["experiment_id"],
        "model": config["model"],
        "selected_calibration": selected_method,
        **calibration_metrics(test_binary, test_probabilities),
    }


def main() -> int:
    experiments = load_paths()["experiments"]
    latest_path = experiments / "latest_run.json"
    if not latest_path.exists():
        raise FileNotFoundError("No baseline latest_run.json is available; run script 05 first")
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    run_root = Path(latest["run_path"])
    rows = []
    for experiment_dir in sorted(path for path in run_root.iterdir() if path.is_dir()):
        result = _run_experiment(experiment_dir)
        if result is not None:
            rows.append(result)
    pd.DataFrame(rows).to_csv(run_root / "calibration_selective_summary.csv", index=False)
    print(f"Wrote calibration and selective-prediction outputs for {len(rows)} experiments")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
