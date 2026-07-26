from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


CONTRIBUTION_COLUMNS = [
    "record_id",
    "feature",
    "tfidf_value",
    "coefficient_unreliable",
    "contribution",
    "direction",
    "direction_rank",
]

SUMMARY_COLUMNS = [
    "record_id",
    "model_prediction",
    "decision_score_unreliable",
    "intercept_unreliable",
    "all_feature_contribution_sum",
    "reconstructed_decision_score",
    "reconstruction_error",
    "nonzero_tfidf_features",
    "emitted_explanation_features",
]


def _linear_tfidf_components(model: Pipeline) -> tuple[object, object, np.ndarray, float]:
    if not isinstance(model, Pipeline):
        raise TypeError("model must be a fitted sklearn Pipeline")
    if "tfidf" not in model.named_steps or "clf" not in model.named_steps:
        raise ValueError("model must contain fitted 'tfidf' and 'clf' steps")

    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["clf"]
    if not hasattr(vectorizer, "vocabulary_") or not hasattr(classifier, "coef_"):
        raise ValueError("model must be fitted and use a linear classifier")

    classes = list(classifier.classes_)
    if len(classes) != 2 or "unreliable" not in classes:
        raise ValueError("classifier must be binary and include the 'unreliable' class")
    unreliable_index = classes.index("unreliable")
    coefficients = np.asarray(classifier.coef_[0], dtype=float)
    intercept = float(np.asarray(classifier.intercept_, dtype=float)[0])
    if unreliable_index == 0:
        coefficients = -coefficients
        intercept = -intercept
    return vectorizer, classifier, coefficients, intercept


def explain_linear_tfidf(
    model: Pipeline,
    texts: Sequence[object] | pd.Series,
    record_ids: Sequence[object] | pd.Series | None = None,
    top_per_direction: int | None = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Explain binary linear TF-IDF decisions with signed additive contributions.

    The unreliable-oriented decision score is reconstructed exactly as
    ``intercept + sum(tfidf_value * coefficient)``. The returned contribution
    table may be limited to the strongest features in each direction, while the
    summary always uses every nonzero feature for its reconstruction.
    """

    if top_per_direction is not None and top_per_direction <= 0:
        raise ValueError("top_per_direction must be positive or None")
    text_values = pd.Series(texts, dtype="object").fillna("").astype(str).tolist()
    if record_ids is None:
        identifiers = [str(index) for index in range(len(text_values))]
    else:
        identifiers = pd.Series(record_ids, dtype="object").astype(str).tolist()
    if len(identifiers) != len(text_values):
        raise ValueError("record_ids and texts must have the same length")

    vectorizer, classifier, coefficients, intercept = _linear_tfidf_components(model)
    feature_names = np.asarray(vectorizer.get_feature_names_out(), dtype=object)
    matrix = vectorizer.transform(text_values).tocsr()
    raw_scores = np.asarray(classifier.decision_function(matrix), dtype=float).reshape(-1)
    if list(classifier.classes_).index("unreliable") == 0:
        raw_scores = -raw_scores
    predictions = np.asarray(model.predict(text_values), dtype=object)

    contribution_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for row_index, record_id in enumerate(identifiers):
        start, end = matrix.indptr[row_index], matrix.indptr[row_index + 1]
        feature_indices = matrix.indices[start:end]
        tfidf_values = matrix.data[start:end].astype(float, copy=False)
        feature_contributions = tfidf_values * coefficients[feature_indices]
        contribution_sum = float(feature_contributions.sum())
        reconstructed = intercept + contribution_sum
        reconstruction_error = abs(reconstructed - float(raw_scores[row_index]))
        if reconstruction_error > 1e-8:
            raise RuntimeError(
                f"Unable to reconstruct decision score for record {record_id}: "
                f"absolute error {reconstruction_error}"
            )

        entries = [
            {
                "feature": str(feature_names[feature_index]),
                "tfidf_value": float(tfidf_value),
                "coefficient_unreliable": float(coefficients[feature_index]),
                "contribution": float(contribution),
            }
            for feature_index, tfidf_value, contribution in zip(
                feature_indices,
                tfidf_values,
                feature_contributions,
                strict=True,
            )
            if contribution != 0
        ]
        positive = sorted(
            (entry for entry in entries if entry["contribution"] > 0),
            key=lambda entry: (-float(entry["contribution"]), str(entry["feature"])),
        )
        negative = sorted(
            (entry for entry in entries if entry["contribution"] < 0),
            key=lambda entry: (float(entry["contribution"]), str(entry["feature"])),
        )
        if top_per_direction is not None:
            positive = positive[:top_per_direction]
            negative = negative[:top_per_direction]
        for direction, selected in (
            ("toward_unreliable", positive),
            ("toward_reliable", negative),
        ):
            for rank, entry in enumerate(selected, start=1):
                contribution_rows.append(
                    {
                        "record_id": record_id,
                        **entry,
                        "direction": direction,
                        "direction_rank": rank,
                    }
                )

        summary_rows.append(
            {
                "record_id": record_id,
                "model_prediction": str(predictions[row_index]),
                "decision_score_unreliable": float(raw_scores[row_index]),
                "intercept_unreliable": intercept,
                "all_feature_contribution_sum": contribution_sum,
                "reconstructed_decision_score": reconstructed,
                "reconstruction_error": reconstruction_error,
                "nonzero_tfidf_features": int(len(feature_indices)),
                "emitted_explanation_features": len(positive) + len(negative),
            }
        )

    return (
        pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS),
        pd.DataFrame(contribution_rows, columns=CONTRIBUTION_COLUMNS),
    )


def _top_contribution_map(
    frame: pd.DataFrame,
    top_k: int,
) -> dict[str, float]:
    selected = frame.assign(_magnitude=frame["contribution"].abs()).sort_values(
        ["_magnitude", "feature"],
        ascending=[False, True],
    )
    selected = selected.drop_duplicates("feature").head(top_k)
    return {
        str(row.feature): float(row.contribution)
        for row in selected.itertuples(index=False)
    }


def compare_standard_controlled_explanations(
    contributions: pd.DataFrame,
    summaries: pd.DataFrame,
    top_k: int = 10,
) -> pd.DataFrame:
    """Compare same-record local explanations across standard and controlled models."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    required_summary = {
        "dataset",
        "model",
        "split_name",
        "record_id",
        "model_prediction",
        "decision_score_unreliable",
    }
    required_contributions = {
        "dataset",
        "model",
        "split_name",
        "record_id",
        "feature",
        "contribution",
    }
    if missing := required_summary.difference(summaries.columns):
        raise ValueError(f"summaries are missing required columns: {sorted(missing)}")
    if missing := required_contributions.difference(contributions.columns):
        raise ValueError(
            f"contributions are missing required columns: {sorted(missing)}"
        )

    standard = summaries[summaries["split_name"].eq("standard")].copy()
    controlled = summaries[summaries["split_name"].eq("controlled")].copy()
    paired = standard.merge(
        controlled,
        on=["dataset", "model", "record_id"],
        suffixes=("_standard", "_controlled"),
        validate="one_to_one",
    )
    lookup = {
        (str(dataset), str(model), str(split_name), str(record_id)): group
        for (dataset, model, split_name, record_id), group in contributions.groupby(
            ["dataset", "model", "split_name", "record_id"],
            sort=False,
        )
    }

    rows: list[dict[str, object]] = []
    empty = contributions.iloc[0:0]
    for row in paired.itertuples(index=False):
        key = (str(row.dataset), str(row.model), "standard", str(row.record_id))
        standard_map = _top_contribution_map(lookup.get(key, empty), top_k)
        key = (str(row.dataset), str(row.model), "controlled", str(row.record_id))
        controlled_map = _top_contribution_map(lookup.get(key, empty), top_k)
        standard_features = set(standard_map)
        controlled_features = set(controlled_map)
        union = standard_features | controlled_features
        overlap = standard_features & controlled_features
        feature_jaccard = len(overlap) / len(union) if union else 1.0
        sign_agreement = (
            sum(np.sign(standard_map[feature]) == np.sign(controlled_map[feature]) for feature in overlap)
            / len(overlap)
            if overlap
            else np.nan
        )
        ordered_union = sorted(union)
        standard_vector = np.array(
            [standard_map.get(feature, 0.0) for feature in ordered_union],
            dtype=float,
        )
        controlled_vector = np.array(
            [controlled_map.get(feature, 0.0) for feature in ordered_union],
            dtype=float,
        )
        denominator = float(
            np.linalg.norm(standard_vector) * np.linalg.norm(controlled_vector)
        )
        contribution_cosine = (
            float(np.dot(standard_vector, controlled_vector) / denominator)
            if denominator
            else np.nan
        )
        result = {
            "dataset": row.dataset,
            "model": row.model,
            "record_id": row.record_id,
            "top_k": top_k,
            "standard_features": len(standard_features),
            "controlled_features": len(controlled_features),
            "overlapping_features": len(overlap),
            "feature_jaccard": feature_jaccard,
            "overlap_sign_agreement": sign_agreement,
            "top_contribution_cosine": contribution_cosine,
            "standard_prediction": row.model_prediction_standard,
            "controlled_prediction": row.model_prediction_controlled,
            "prediction_agreement": (
                row.model_prediction_standard == row.model_prediction_controlled
            ),
            "standard_decision_score": row.decision_score_unreliable_standard,
            "controlled_decision_score": row.decision_score_unreliable_controlled,
        }
        if hasattr(row, "record_manifest_split_standard"):
            result["standard_record_manifest_split"] = (
                row.record_manifest_split_standard
            )
            result["controlled_record_manifest_split"] = (
                row.record_manifest_split_controlled
            )
            result["both_models_held_out"] = (
                row.record_manifest_split_standard == "test"
                and row.record_manifest_split_controlled == "test"
            )
        for optional in ("case_id", "selection_reasons"):
            standard_name = f"{optional}_standard"
            if hasattr(row, standard_name):
                result[optional] = getattr(row, standard_name)
        rows.append(result)
    return pd.DataFrame(rows)
