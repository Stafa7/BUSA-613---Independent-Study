"""Quantitative covariate-shift diagnostics for cross-dataset transfer.

The helpers in this module are deliberately independent of fitted classifiers.
They compare the exact source training cohort with the post-exclusion target test
cohort and make no claim that the two datasets' labels are interchangeable.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from health_misinfo.evaluation.metrics import classification_metrics


SOURCE_COHORT = "source_train"
TARGET_COHORT = "target_test_after_domain_exclusion"
MISSING_CATEGORY = "(missing)"
DEGRADATION_METRICS = ("macro_f1", "unreliable_f1", "unreliable_pr_auc")


@dataclass(frozen=True)
class TransferShiftArtifacts:
    """Serializable in-memory outputs for one transfer direction."""

    metadata: dict[str, Any]
    tables: dict[str, pd.DataFrame]
    markdown: str


@dataclass(frozen=True)
class BaselineComparatorSpec:
    """Exact baseline artifact expected for a transfer comparison."""

    comparator_kind: str
    comparator_dataset: str
    baseline_experiment_id: str | None
    baseline_model: str | None
    text_column: str | None
    same_evaluation_records: bool
    is_primary_comparator: bool
    definition: str
    unavailable_reason: str = ""


def _texts(values: Iterable[object]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value is None or (
            not isinstance(value, (list, tuple, dict)) and pd.isna(value)
        ):
            result.append("")
        else:
            result.append(str(value))
    return result


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else math.nan


def jensen_shannon_divergence(
    left: Sequence[float] | np.ndarray,
    right: Sequence[float] | np.ndarray,
) -> float:
    """Return symmetric Jensen-Shannon divergence in bits (bounded by 0 and 1)."""

    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    if (
        left_array.ndim != 1
        or right_array.ndim != 1
        or left_array.shape != right_array.shape
        or left_array.size == 0
    ):
        raise ValueError("Inputs must be non-empty, equally sized 1D vectors.")
    if (
        np.any(left_array < 0)
        or np.any(right_array < 0)
        or left_array.sum() <= 0
        or right_array.sum() <= 0
    ):
        raise ValueError("Inputs must be non-negative with positive total mass.")
    left_probability = left_array / left_array.sum()
    right_probability = right_array / right_array.sum()
    midpoint = (left_probability + right_probability) / 2

    def _kl(probability: np.ndarray) -> float:
        positive = probability > 0
        return float(
            np.sum(
                probability[positive]
                * np.log2(probability[positive] / midpoint[positive])
            )
        )

    return (_kl(left_probability) + _kl(right_probability)) / 2


def total_variation_distance(
    left: Sequence[float] | np.ndarray,
    right: Sequence[float] | np.ndarray,
) -> float:
    """Return total variation distance between two count/probability vectors."""

    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    if (
        left_array.ndim != 1
        or right_array.ndim != 1
        or left_array.shape != right_array.shape
        or left_array.size == 0
    ):
        raise ValueError("Inputs must be non-empty, equally sized 1D vectors.")
    if (
        np.any(left_array < 0)
        or np.any(right_array < 0)
        or left_array.sum() <= 0
        or right_array.sum() <= 0
    ):
        raise ValueError("Inputs must be non-negative with positive total mass.")
    left_probability = left_array / left_array.sum()
    right_probability = right_array / right_array.sum()
    return float(np.abs(left_probability - right_probability).sum() / 2)


def kolmogorov_smirnov_statistic(
    left: Sequence[float] | np.ndarray,
    right: Sequence[float] | np.ndarray,
) -> float:
    """Return the descriptive two-sample KS statistic without a p-value."""

    left_array = np.sort(np.asarray(left, dtype=float))
    right_array = np.sort(np.asarray(right, dtype=float))
    left_array = left_array[np.isfinite(left_array)]
    right_array = right_array[np.isfinite(right_array)]
    if not len(left_array) or not len(right_array):
        return math.nan
    support = np.unique(np.concatenate([left_array, right_array]))
    left_cdf = np.searchsorted(left_array, support, side="right") / len(left_array)
    right_cdf = (
        np.searchsorted(right_array, support, side="right") / len(right_array)
    )
    return float(np.max(np.abs(left_cdf - right_cdf)))


def standardized_mean_difference(
    source: Sequence[float] | np.ndarray,
    target: Sequence[float] | np.ndarray,
) -> float:
    """Return target-minus-source Cohen's d using the pooled sample SD."""

    source_array = np.asarray(source, dtype=float)
    target_array = np.asarray(target, dtype=float)
    source_array = source_array[np.isfinite(source_array)]
    target_array = target_array[np.isfinite(target_array)]
    if len(source_array) < 2 or len(target_array) < 2:
        return math.nan
    pooled_variance = (
        (len(source_array) - 1) * np.var(source_array, ddof=1)
        + (len(target_array) - 1) * np.var(target_array, ddof=1)
    ) / (len(source_array) + len(target_array) - 2)
    if pooled_variance == 0:
        return (
            0.0
            if float(np.mean(source_array)) == float(np.mean(target_array))
            else math.nan
        )
    return float(
        (np.mean(target_array) - np.mean(source_array))
        / math.sqrt(pooled_variance)
    )


def word_counts(values: Iterable[object]) -> np.ndarray:
    """Count tokens with scikit-learn's standard two-character token rule."""

    analyzer = CountVectorizer(strip_accents="unicode").build_analyzer()
    return np.asarray([len(analyzer(text)) for text in _texts(values)], dtype=int)


def document_length_diagnostics(
    source_texts: Iterable[object],
    target_texts: Iterable[object],
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Return cohort length profiles and source-to-target shift metrics."""

    source = word_counts(source_texts)
    target = word_counts(target_texts)

    def _profile(cohort: str, counts: np.ndarray) -> dict[str, Any]:
        if not len(counts):
            return {"cohort": cohort, "records": 0}
        return {
            "cohort": cohort,
            "records": len(counts),
            "empty_document_share": float(np.mean(counts == 0)),
            "mean_tokens": float(np.mean(counts)),
            "standard_deviation_tokens": (
                float(np.std(counts, ddof=1)) if len(counts) > 1 else 0.0
            ),
            "minimum_tokens": int(np.min(counts)),
            "p10_tokens": float(np.quantile(counts, 0.10)),
            "p25_tokens": float(np.quantile(counts, 0.25)),
            "median_tokens": float(np.median(counts)),
            "p75_tokens": float(np.quantile(counts, 0.75)),
            "p90_tokens": float(np.quantile(counts, 0.90)),
            "maximum_tokens": int(np.max(counts)),
        }

    profiles = pd.DataFrame(
        [_profile(SOURCE_COHORT, source), _profile(TARGET_COHORT, target)]
    )
    source_mean = float(np.mean(source)) if len(source) else math.nan
    target_mean = float(np.mean(target)) if len(target) else math.nan
    source_median = float(np.median(source)) if len(source) else math.nan
    target_median = float(np.median(target)) if len(target) else math.nan
    shift = {
        "target_minus_source_mean_tokens": target_mean - source_mean,
        "target_to_source_mean_ratio": _ratio(target_mean, source_mean),
        "target_minus_source_median_tokens": target_median - source_median,
        "target_to_source_median_ratio": _ratio(target_median, source_median),
        "standardized_mean_difference_target_minus_source": (
            standardized_mean_difference(source, target)
        ),
        "kolmogorov_smirnov_statistic": kolmogorov_smirnov_statistic(
            source, target
        ),
    }
    return profiles, shift


def _categories(values: Iterable[object]) -> pd.Series:
    categories = pd.Series(_texts(values), dtype="object").str.strip()
    return categories.mask(categories.eq(""), MISSING_CATEGORY)


def categorical_shift_diagnostics(
    source_values: Iterable[object],
    target_values: Iterable[object],
    *,
    field: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Align categorical mixes and quantify prevalence/composition differences."""

    source = _categories(source_values)
    target = _categories(target_values)
    if source.empty or target.empty:
        raise ValueError(f"Both cohorts must be non-empty for {field!r}.")
    source_counts = source.value_counts(sort=False)
    target_counts = target.value_counts(sort=False)
    category_names = sorted(set(source_counts.index) | set(target_counts.index))
    source_aligned = source_counts.reindex(category_names, fill_value=0)
    target_aligned = target_counts.reindex(category_names, fill_value=0)
    source_share = source_aligned / len(source)
    target_share = target_aligned / len(target)
    distribution = pd.DataFrame(
        {
            "field": field,
            "category": category_names,
            "source_count": source_aligned.to_numpy(dtype=int),
            "source_share": source_share.to_numpy(dtype=float),
            "target_count": target_aligned.to_numpy(dtype=int),
            "target_share": target_share.to_numpy(dtype=float),
        }
    )
    distribution["target_minus_source_share"] = (
        distribution["target_share"] - distribution["source_share"]
    )

    source_present = set(source_counts.index)
    target_present = set(target_counts.index)
    shared = source_present & target_present
    source_hhi = float(np.square(source_share).sum())
    target_hhi = float(np.square(target_share).sum())
    summary: dict[str, Any] = {
        "field": field,
        "source_unique_categories": len(source_present),
        "target_unique_categories": len(target_present),
        "shared_categories": len(shared),
        "category_jaccard": _ratio(
            len(shared), len(source_present | target_present)
        ),
        "target_record_share_in_source_categories": float(
            target.isin(source_present).mean()
        ),
        "source_record_share_in_target_categories": float(
            source.isin(target_present).mean()
        ),
        "total_variation_distance": total_variation_distance(
            source_share, target_share
        ),
        "jensen_shannon_divergence_bits": jensen_shannon_divergence(
            source_share, target_share
        ),
        "source_top1_share": float(source_counts.max() / len(source)),
        "target_top1_share": float(target_counts.max() / len(target)),
        "source_top5_share": float(source_counts.nlargest(5).sum() / len(source)),
        "target_top5_share": float(target_counts.nlargest(5).sum() / len(target)),
        "source_hhi": source_hhi,
        "target_hhi": target_hhi,
        "source_effective_category_count": _ratio(1, source_hhi),
        "target_effective_category_count": _ratio(1, target_hhi),
        "source_missing_share": float(source.eq(MISSING_CATEGORY).mean()),
        "target_missing_share": float(target.eq(MISSING_CATEGORY).mean()),
    }
    if field == "harmonized_label":
        source_prevalence = float(source.eq("unreliable").mean())
        target_prevalence = float(target.eq("unreliable").mean())
        summary.update(
            {
                "source_unreliable_prevalence": source_prevalence,
                "target_unreliable_prevalence": target_prevalence,
                "target_minus_source_unreliable_prevalence": (
                    target_prevalence - source_prevalence
                ),
            }
        )
    return distribution, summary


def vocabulary_shift_diagnostics(
    source_texts: Iterable[object],
    target_texts: Iterable[object],
    *,
    max_features: int = 20_000,
    minimum_term_count: int = 2,
    top_terms_per_direction: int = 25,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Measure overlap/divergence in a deterministic capped pooled unigram space."""

    source = _texts(source_texts)
    target = _texts(target_texts)
    if not source or not target:
        raise ValueError("Both cohorts must be non-empty.")
    if min(max_features, minimum_term_count, top_terms_per_direction) < 1:
        raise ValueError("Vocabulary parameters must be positive.")
    vectorizer = CountVectorizer(
        strip_accents="unicode",
        token_pattern=r"(?u)\b\w\w+\b",
        max_features=max_features,
    )
    try:
        matrix = vectorizer.fit_transform(source + target)
    except ValueError as error:
        if "empty vocabulary" not in str(error).lower():
            raise
        return pd.DataFrame(), {
            "status": "empty_vocabulary",
            "retained_vocabulary_terms": 0,
        }
    source_counts = np.asarray(matrix[: len(source)].sum(axis=0)).ravel()
    target_counts = np.asarray(matrix[len(source) :].sum(axis=0)).ravel()
    terms = vectorizer.get_feature_names_out()
    source_total = float(source_counts.sum())
    target_total = float(target_counts.sum())
    if not source_total or not target_total:
        return pd.DataFrame(), {
            "status": "one_cohort_has_no_retained_tokens",
            "retained_vocabulary_terms": len(terms),
        }

    source_active = source_counts >= minimum_term_count
    target_active = target_counts >= minimum_term_count
    shared_active = source_active & target_active
    active_union = source_active | target_active
    source_share = source_counts / source_total
    target_share = target_counts / target_total
    share_delta = target_share - source_share
    smoothing = 0.5
    source_smoothed = (source_counts + smoothing) / (
        source_total + smoothing * len(terms)
    )
    target_smoothed = (target_counts + smoothing) / (
        target_total + smoothing * len(terms)
    )
    log_ratio = np.log2(target_smoothed / source_smoothed)
    eligible = np.flatnonzero(
        source_counts + target_counts >= minimum_term_count
    )
    source_ranked = sorted(
        eligible, key=lambda index: (share_delta[index], terms[index])
    )[:top_terms_per_direction]
    target_ranked = sorted(
        eligible, key=lambda index: (-share_delta[index], terms[index])
    )[:top_terms_per_direction]
    rows: list[dict[str, Any]] = []
    for enrichment, indices in (
        ("source_enriched", source_ranked),
        ("target_enriched", target_ranked),
    ):
        for rank, index in enumerate(indices, start=1):
            rows.append(
                {
                    "enrichment": enrichment,
                    "rank": rank,
                    "term": terms[index],
                    "source_count": int(source_counts[index]),
                    "source_share": float(source_share[index]),
                    "target_count": int(target_counts[index]),
                    "target_share": float(target_share[index]),
                    "target_minus_source_share": float(share_delta[index]),
                    "target_to_source_log2_ratio_smoothed": float(
                        log_ratio[index]
                    ),
                }
            )
    summary = {
        "status": "ok",
        "retained_vocabulary_terms": len(terms),
        "source_retained_tokens": int(source_total),
        "target_retained_tokens": int(target_total),
        "source_active_types": int(source_active.sum()),
        "target_active_types": int(target_active.sum()),
        "shared_active_types": int(shared_active.sum()),
        "active_type_jaccard": _ratio(
            int(shared_active.sum()), int(active_union.sum())
        ),
        "target_active_type_coverage_by_source": _ratio(
            int(shared_active.sum()), int(target_active.sum())
        ),
        "target_token_coverage_by_source_active_types": float(
            target_counts[source_active].sum() / target_total
        ),
        "source_token_coverage_by_target_active_types": float(
            source_counts[target_active].sum() / source_total
        ),
        "jensen_shannon_divergence_bits_retained_vocabulary": (
            jensen_shannon_divergence(source_counts, target_counts)
        ),
        "maximum_features": max_features,
        "minimum_active_term_count": minimum_term_count,
    }
    return pd.DataFrame(rows), summary


def _topic_mix(
    weights: np.ndarray,
    *,
    cohort: str,
    topic_ids: list[str],
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    """Summarize soft and dominant topic assignments for one cohort."""

    row_totals = weights.sum(axis=1)
    assigned = row_totals > 0
    normalized = np.zeros_like(weights)
    normalized[assigned] = weights[assigned] / row_totals[assigned, None]
    dominant = np.full(len(weights), -1, dtype=int)
    dominant[assigned] = np.argmax(normalized[assigned], axis=1)
    dominant_share = np.zeros(len(weights), dtype=float)
    dominant_share[assigned] = np.max(normalized[assigned], axis=1)
    mean_weights = normalized.sum(axis=0) / len(weights)
    rows = [
        {
            "cohort": cohort,
            "topic_id": topic_id,
            "mean_weight_share": float(mean_weights[index]),
            "dominant_document_count": int((dominant == index).sum()),
            "dominant_document_share": float(np.mean(dominant == index)),
        }
        for index, topic_id in enumerate(topic_ids)
    ]
    unassigned_share = float(np.mean(~assigned))
    rows.append(
        {
            "cohort": cohort,
            "topic_id": "unassigned",
            "mean_weight_share": unassigned_share,
            "dominant_document_count": int((~assigned).sum()),
            "dominant_document_share": unassigned_share,
        }
    )
    return (
        rows,
        np.append(mean_weights, unassigned_share),
        dominant,
        dominant_share,
    )


def topic_proxy_diagnostics(
    source_texts: Iterable[object],
    target_texts: Iterable[object],
    *,
    source_ids: Iterable[object] | None = None,
    target_ids: Iterable[object] | None = None,
    random_seed: int = 613,
    requested_topics: int = 10,
    max_features: int = 5_000,
    top_terms_per_topic: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Compare pooled TF-IDF/NMF topic mixtures without using labels.

    NMF components are an interpretable, deterministic topic proxy for
    covariate-shift diagnosis. They are not treated as validated narratives.
    """

    source = _texts(source_texts)
    target = _texts(target_texts)
    if not source or not target:
        raise ValueError("Both cohorts must be non-empty.")
    if min(requested_topics, max_features, top_terms_per_topic) < 1:
        raise ValueError("Topic parameters must be positive.")
    source_record_ids = (
        _texts(source_ids)
        if source_ids is not None
        else [str(index) for index in range(len(source))]
    )
    target_record_ids = (
        _texts(target_ids)
        if target_ids is not None
        else [str(index) for index in range(len(target))]
    )
    if len(source_record_ids) != len(source) or len(target_record_ids) != len(
        target
    ):
        raise ValueError("Record IDs must align with their text cohorts.")

    # The fallbacks make the helper well-defined for very small or repetitive
    # samples while recording which vocabulary rule was ultimately used.
    attempts = (
        (
            "english_stopwords_min_df_2_max_df_0.95",
            {"stop_words": "english", "min_df": 2, "max_df": 0.95},
        ),
        (
            "english_stopwords_min_df_1",
            {"stop_words": "english", "min_df": 1, "max_df": 1.0},
        ),
        (
            "no_stopwords_min_df_1",
            {"stop_words": None, "min_df": 1, "max_df": 1.0},
        ),
    )
    matrix = None
    vectorizer = None
    vectorizer_strategy = ""
    for strategy, parameters in attempts:
        candidate = TfidfVectorizer(
            strip_accents="unicode",
            token_pattern=r"(?u)\b\w\w+\b",
            ngram_range=(1, 2),
            sublinear_tf=True,
            max_features=max_features,
            **parameters,
        )
        try:
            candidate_matrix = candidate.fit_transform(source + target)
        except ValueError:
            continue
        if candidate_matrix.shape[1] and candidate_matrix.nnz:
            matrix = candidate_matrix
            vectorizer = candidate
            vectorizer_strategy = strategy
            break

    if matrix is None or vectorizer is None:
        assignments = pd.DataFrame(
            {
                "cohort": [SOURCE_COHORT] * len(source)
                + [TARGET_COHORT] * len(target),
                "record_id": source_record_ids + target_record_ids,
                "dominant_topic": ["unassigned"] * (len(source) + len(target)),
                "dominant_topic_weight_share": [0.0]
                * (len(source) + len(target)),
            }
        )
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            assignments,
            {
                "status": "empty_vocabulary",
                "method": "pooled_tfidf_nmf",
                "fitted_topics": 0,
            },
        )

    fitted_topics = min(requested_topics, matrix.shape[0], matrix.shape[1])
    model = NMF(
        n_components=fitted_topics,
        init="nndsvda",
        random_state=random_seed,
        solver="cd",
        max_iter=400,
    )
    weights = model.fit_transform(matrix)
    topic_ids = [f"topic_{index:02d}" for index in range(fitted_topics)]
    feature_names = vectorizer.get_feature_names_out()
    top_term_rows: list[dict[str, Any]] = []
    short_labels: dict[str, str] = {}
    for index, topic_id in enumerate(topic_ids):
        ranked = np.argsort(model.components_[index])[::-1][
            : min(top_terms_per_topic, len(feature_names))
        ]
        terms = [str(feature_names[position]) for position in ranked]
        short_labels[topic_id] = ", ".join(terms[:5])
        top_term_rows.extend(
            {
                "topic_id": topic_id,
                "rank": rank,
                "term": str(feature_names[position]),
                "component_weight": float(model.components_[index, position]),
            }
            for rank, position in enumerate(ranked, start=1)
        )

    source_rows, source_mix, source_dominant, source_dominant_share = _topic_mix(
        weights[: len(source)], cohort=SOURCE_COHORT, topic_ids=topic_ids
    )
    target_rows, target_mix, target_dominant, target_dominant_share = _topic_mix(
        weights[len(source) :], cohort=TARGET_COHORT, topic_ids=topic_ids
    )
    distribution = pd.DataFrame(source_rows + target_rows)
    source_shares = distribution[
        distribution["cohort"].eq(SOURCE_COHORT)
    ].set_index("topic_id")["mean_weight_share"]
    target_shares = distribution[
        distribution["cohort"].eq(TARGET_COHORT)
    ].set_index("topic_id")["mean_weight_share"]
    delta = target_shares - source_shares
    distribution["target_minus_source_mean_weight_share"] = distribution[
        "topic_id"
    ].map(delta)

    def _assignment_rows(
        cohort: str,
        record_ids: list[str],
        dominant: np.ndarray,
        shares: np.ndarray,
    ) -> list[dict[str, Any]]:
        return [
            {
                "cohort": cohort,
                "record_id": record_id,
                "dominant_topic": (
                    topic_ids[topic_index]
                    if topic_index >= 0
                    else "unassigned"
                ),
                "dominant_topic_weight_share": float(share),
            }
            for record_id, topic_index, share in zip(
                record_ids, dominant, shares
            )
        ]

    assignments = pd.DataFrame(
        _assignment_rows(
            SOURCE_COHORT,
            source_record_ids,
            source_dominant,
            source_dominant_share,
        )
        + _assignment_rows(
            TARGET_COHORT,
            target_record_ids,
            target_dominant,
            target_dominant_share,
        )
    )
    topic_delta = delta.drop(index="unassigned", errors="ignore")
    target_enriched = str(topic_delta.idxmax())
    source_enriched = str(topic_delta.idxmin())
    summary = {
        "status": "ok",
        "method": "pooled_tfidf_nmf",
        "vectorizer_strategy": vectorizer_strategy,
        "requested_topics": requested_topics,
        "fitted_topics": fitted_topics,
        "retained_features": matrix.shape[1],
        "random_seed": random_seed,
        "jensen_shannon_divergence_bits_mean_topic_weights": (
            jensen_shannon_divergence(source_mix, target_mix)
        ),
        "total_variation_distance_mean_topic_weights": (
            total_variation_distance(source_mix, target_mix)
        ),
        "most_target_enriched_topic": target_enriched,
        "most_target_enriched_topic_delta": float(topic_delta[target_enriched]),
        "most_target_enriched_topic_terms": short_labels[target_enriched],
        "most_source_enriched_topic": source_enriched,
        "most_source_enriched_topic_delta": float(topic_delta[source_enriched]),
        "most_source_enriched_topic_terms": short_labels[source_enriched],
    }
    return pd.DataFrame(top_term_rows), distribution, assignments, summary


def baseline_comparator_spec(
    *,
    source_dataset: str,
    target_dataset: str,
    text_representation: str,
    transfer_model: str,
    comparator_kind: str,
) -> BaselineComparatorSpec:
    """Resolve an exact, non-fuzzy in-domain baseline artifact.

    ``target_in_domain_same_records`` is primary: it compares transfer with a
    target-trained baseline on the exact target records retained after domain
    exclusion. ``source_in_domain_standard_test`` is secondary and represents
    the conventional drop from source-domain test performance to transfer, but
    it necessarily changes the evaluation records and label construct.
    """

    if comparator_kind == "target_in_domain_same_records":
        comparator_dataset = target_dataset
        same_records = True
        primary = True
        definition = (
            "Target-trained standard-split baseline with the same estimator "
            "family and text representation, recomputed on the exact "
            "post-domain-exclusion target records used by transfer. "
            "Hyperparameters are independently selected on target validation."
        )
    elif comparator_kind == "source_in_domain_standard_test":
        comparator_dataset = source_dataset
        same_records = False
        primary = False
        definition = (
            "Source-trained standard-split baseline with the same estimator "
            "family and text representation on the source standard test set. "
            "This conventional transfer drop changes evaluation records and "
            "label construct, so it is descriptive rather than causal."
        )
    else:
        raise ValueError(f"Unknown comparator kind: {comparator_kind}")

    baseline_model: str | None = None
    text_column: str | None = None
    experiment_id: str | None = None
    reason = ""
    if text_representation == "full_text" and transfer_model in {
        "logistic_regression",
        "linear_svm",
    }:
        baseline_model = transfer_model
        text_column = "model_text"
        experiment_id = (
            f"{comparator_dataset}_standard_{baseline_model}_{text_column}"
        )
    elif (
        text_representation == "title_only"
        and transfer_model == "logistic_regression"
    ):
        baseline_model = "title_only_logistic"
        text_column = "title"
        experiment_id = (
            f"{comparator_dataset}_standard_{baseline_model}_{text_column}"
        )
    elif (
        text_representation == "title_only"
        and transfer_model == "linear_svm"
    ):
        reason = "same_family_title_only_baseline_not_in_baseline_matrix"
    elif text_representation not in {"full_text", "title_only"}:
        reason = "unsupported_text_representation"
    else:
        reason = "unsupported_transfer_model"

    return BaselineComparatorSpec(
        comparator_kind=comparator_kind,
        comparator_dataset=comparator_dataset,
        baseline_experiment_id=experiment_id,
        baseline_model=baseline_model,
        text_column=text_column,
        same_evaluation_records=same_records,
        is_primary_comparator=primary,
        definition=definition,
        unavailable_reason=reason,
    )


def _degradation_unavailable_rows(
    base: dict[str, Any],
    *,
    reason: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                **base,
                "metric": metric,
                "comparison_status": "unavailable",
                "unavailable_reason": reason,
                "baseline_full_standard_value": math.nan,
                "in_domain_comparator_value": math.nan,
                "transfer_value": math.nan,
                "absolute_delta_transfer_minus_in_domain": math.nan,
                "relative_delta_transfer_minus_in_domain": math.nan,
                "absolute_degradation_in_domain_minus_transfer": math.nan,
                "relative_degradation_in_domain_minus_transfer": math.nan,
                "relative_delta_status": "unavailable",
                "relative_delta_unavailable_reason": reason,
            }
            for metric in DEGRADATION_METRICS
        ]
    )


def transfer_degradation_rows(
    *,
    transfer_experiment_id: str,
    source_dataset: str,
    target_dataset: str,
    text_representation: str,
    transfer_model: str,
    transfer_predictions: pd.DataFrame,
    comparator: BaselineComparatorSpec,
    baseline_summary: pd.DataFrame,
    baseline_config: dict[str, Any] | None,
    baseline_predictions: pd.DataFrame | None,
    baseline_config_unavailable_reason: str = "",
    baseline_predictions_unavailable_reason: str = "",
) -> pd.DataFrame:
    """Compare a transfer result with a validated, compatible in-domain baseline."""

    base: dict[str, Any] = {
        "transfer_experiment_id": transfer_experiment_id,
        "source_dataset": source_dataset,
        "target_dataset": target_dataset,
        "text_representation": text_representation,
        "transfer_model": transfer_model,
        "comparator_kind": comparator.comparator_kind,
        "is_primary_comparator": comparator.is_primary_comparator,
        "comparator_definition": comparator.definition,
        "comparator_dataset": comparator.comparator_dataset,
        "baseline_experiment_id": comparator.baseline_experiment_id or "",
        "same_evaluation_records": comparator.same_evaluation_records,
        "transfer_records": len(transfer_predictions),
        "baseline_full_standard_records": math.nan,
        "comparison_records": math.nan,
    }
    if comparator.unavailable_reason:
        return _degradation_unavailable_rows(
            base, reason=comparator.unavailable_reason
        )
    if baseline_summary.empty:
        return _degradation_unavailable_rows(
            base, reason="latest_baseline_summary_unavailable"
        )
    required_summary_columns = {
        "experiment_id",
        "n",
        *DEGRADATION_METRICS,
    }
    if not required_summary_columns <= set(baseline_summary.columns):
        return _degradation_unavailable_rows(
            base, reason="latest_baseline_summary_schema_incompatible"
        )
    matching_summary = baseline_summary[
        baseline_summary["experiment_id"].eq(
            comparator.baseline_experiment_id
        )
    ]
    if matching_summary.empty:
        return _degradation_unavailable_rows(
            base, reason="exact_baseline_experiment_missing_from_latest_summary"
        )
    if len(matching_summary) != 1:
        return _degradation_unavailable_rows(
            base, reason="duplicate_exact_baseline_experiment_in_latest_summary"
        )
    summary_row = matching_summary.iloc[0]
    base["baseline_full_standard_records"] = int(summary_row["n"])

    if baseline_config_unavailable_reason:
        return _degradation_unavailable_rows(
            base, reason=baseline_config_unavailable_reason
        )
    if baseline_config is None:
        return _degradation_unavailable_rows(
            base, reason="baseline_config_unavailable"
        )
    expected_config = {
        "experiment_id": comparator.baseline_experiment_id,
        "dataset": comparator.comparator_dataset,
        "split_name": "standard",
        "model": comparator.baseline_model,
        "text_column": comparator.text_column,
    }
    mismatched_config_fields = [
        field
        for field, expected in expected_config.items()
        if baseline_config.get(field) != expected
    ]
    if mismatched_config_fields:
        return _degradation_unavailable_rows(
            base,
            reason=(
                "baseline_config_mismatch:"
                + ",".join(sorted(mismatched_config_fields))
            ),
        )

    required_prediction_columns = {
        "record_id",
        "harmonized_label",
        "prediction",
    }
    if not required_prediction_columns <= set(transfer_predictions.columns):
        return _degradation_unavailable_rows(
            base, reason="transfer_prediction_schema_incompatible"
        )
    if transfer_predictions["record_id"].astype(str).duplicated().any():
        return _degradation_unavailable_rows(
            base, reason="duplicate_transfer_prediction_record_ids"
        )
    transfer_score = (
        transfer_predictions["score_unreliable"]
        if "score_unreliable" in transfer_predictions
        else None
    )
    transfer_metrics = classification_metrics(
        transfer_predictions["harmonized_label"],
        transfer_predictions["prediction"],
        transfer_score,
    )

    if comparator.same_evaluation_records:
        if baseline_predictions_unavailable_reason:
            return _degradation_unavailable_rows(
                base, reason=baseline_predictions_unavailable_reason
            )
        if baseline_predictions is None:
            return _degradation_unavailable_rows(
                base, reason="baseline_predictions_unavailable"
            )
        if not required_prediction_columns <= set(baseline_predictions.columns):
            return _degradation_unavailable_rows(
                base, reason="baseline_prediction_schema_incompatible"
            )
        baseline_ids = baseline_predictions["record_id"].astype(str)
        if baseline_ids.duplicated().any():
            return _degradation_unavailable_rows(
                base, reason="duplicate_baseline_prediction_record_ids"
            )
        transfer_ids = set(
            transfer_predictions["record_id"].astype(str)
        )
        missing_ids = transfer_ids - set(baseline_ids)
        if missing_ids:
            base["comparison_records"] = len(transfer_ids) - len(missing_ids)
            return _degradation_unavailable_rows(
                base,
                reason="baseline_predictions_do_not_cover_transfer_record_set",
            )
        baseline_subset = baseline_predictions[
            baseline_ids.isin(transfer_ids)
        ].copy()
        alignment = transfer_predictions[
            ["record_id", "harmonized_label"]
        ].copy()
        alignment["record_id"] = alignment["record_id"].astype(str)
        baseline_subset["record_id"] = baseline_subset["record_id"].astype(str)
        alignment = alignment.merge(
            baseline_subset[
                ["record_id", "harmonized_label"]
            ].rename(columns={"harmonized_label": "baseline_label"}),
            on="record_id",
            how="inner",
            validate="one_to_one",
        )
        if not alignment["harmonized_label"].astype(str).equals(
            alignment["baseline_label"].astype(str)
        ):
            return _degradation_unavailable_rows(
                base, reason="label_mismatch_on_like_for_like_record_set"
            )
        baseline_score = (
            baseline_subset["score_unreliable"]
            if "score_unreliable" in baseline_subset
            else None
        )
        comparator_metrics = classification_metrics(
            baseline_subset["harmonized_label"],
            baseline_subset["prediction"],
            baseline_score,
        )
        base["comparison_records"] = len(baseline_subset)
    else:
        comparator_metrics = {
            metric: summary_row[metric] for metric in DEGRADATION_METRICS
        }
        base["comparison_records"] = int(summary_row["n"])

    rows: list[dict[str, Any]] = []
    both_transfer_classes = (
        transfer_predictions["harmonized_label"].astype(str).nunique() == 2
    )
    for metric in DEGRADATION_METRICS:
        baseline_full_value = summary_row[metric]
        comparator_value = comparator_metrics.get(metric, math.nan)
        transfer_value = transfer_metrics.get(metric, math.nan)
        unavailable_reason = ""
        if pd.isna(comparator_value):
            if (
                metric == "unreliable_pr_auc"
                and comparator.same_evaluation_records
                and not both_transfer_classes
            ):
                unavailable_reason = (
                    "like_for_like_subset_lacks_both_classes_for_pr_auc"
                )
            elif metric == "unreliable_pr_auc" and (
                baseline_predictions is not None
                and "score_unreliable" not in baseline_predictions
            ):
                unavailable_reason = "baseline_score_unavailable_for_pr_auc"
            else:
                unavailable_reason = "in_domain_metric_unavailable"
        elif pd.isna(transfer_value):
            if metric == "unreliable_pr_auc" and not both_transfer_classes:
                unavailable_reason = (
                    "transfer_subset_lacks_both_classes_for_pr_auc"
                )
            elif (
                metric == "unreliable_pr_auc"
                and "score_unreliable" not in transfer_predictions
            ):
                unavailable_reason = "transfer_score_unavailable_for_pr_auc"
            else:
                unavailable_reason = "transfer_metric_unavailable"

        if unavailable_reason:
            rows.append(
                {
                    **base,
                    "metric": metric,
                    "comparison_status": "unavailable",
                    "unavailable_reason": unavailable_reason,
                    "baseline_full_standard_value": baseline_full_value,
                    "in_domain_comparator_value": comparator_value,
                    "transfer_value": transfer_value,
                    "absolute_delta_transfer_minus_in_domain": math.nan,
                    "relative_delta_transfer_minus_in_domain": math.nan,
                    "absolute_degradation_in_domain_minus_transfer": math.nan,
                    "relative_degradation_in_domain_minus_transfer": math.nan,
                    "relative_delta_status": "unavailable",
                    "relative_delta_unavailable_reason": unavailable_reason,
                }
            )
            continue

        comparator_value = float(comparator_value)
        transfer_value = float(transfer_value)
        delta = transfer_value - comparator_value
        degradation = comparator_value - transfer_value
        relative_available = comparator_value != 0
        rows.append(
            {
                **base,
                "metric": metric,
                "comparison_status": "available",
                "unavailable_reason": "",
                "baseline_full_standard_value": float(baseline_full_value),
                "in_domain_comparator_value": comparator_value,
                "transfer_value": transfer_value,
                "absolute_delta_transfer_minus_in_domain": delta,
                "relative_delta_transfer_minus_in_domain": (
                    delta / comparator_value
                    if relative_available
                    else math.nan
                ),
                "absolute_degradation_in_domain_minus_transfer": degradation,
                "relative_degradation_in_domain_minus_transfer": (
                    degradation / comparator_value
                    if relative_available
                    else math.nan
                ),
                "relative_delta_status": (
                    "available" if relative_available else "unavailable"
                ),
                "relative_delta_unavailable_reason": (
                    ""
                    if relative_available
                    else "zero_in_domain_comparator_metric"
                ),
            }
        )
    return pd.DataFrame(rows)


def _cohort_profile(
    frame: pd.DataFrame,
    *,
    direction: str,
    dataset: str,
    cohort: str,
) -> dict[str, Any]:
    labels = _categories(frame["harmonized_label"])
    return {
        "direction": direction,
        "cohort": cohort,
        "dataset": dataset,
        "records": len(frame),
        "reliable_records": int(labels.eq("reliable").sum()),
        "unreliable_records": int(labels.eq("unreliable").sum()),
        "unreliable_prevalence": float(labels.eq("unreliable").mean()),
        "text_units": int(_categories(frame["text_unit"]).nunique()),
        "publishers": int(_categories(frame["publisher_id"]).nunique()),
        "source_domains": int(_categories(frame["source_domain"]).nunique()),
    }


def _long_summary(
    direction: str,
    *,
    exclusion: pd.DataFrame,
    vocabulary: pd.DataFrame,
    length: pd.DataFrame,
    categorical: pd.DataFrame,
    topic: pd.DataFrame,
) -> pd.DataFrame:
    """Flatten selected headline metrics for easy cross-direction comparison."""

    rows: list[dict[str, Any]] = []

    def _add(
        dimension: str,
        component: str,
        record: dict[str, Any],
        metric_names: Sequence[str],
    ) -> None:
        rows.extend(
            {
                "direction": direction,
                "dimension": dimension,
                "component": component,
                "metric": metric,
                "value": record[metric],
            }
            for metric in metric_names
            if metric in record
        )

    for record in vocabulary.to_dict(orient="records"):
        _add(
            "vocabulary",
            str(record["text_representation"]),
            record,
            (
                "active_type_jaccard",
                "target_active_type_coverage_by_source",
                "target_token_coverage_by_source_active_types",
                "jensen_shannon_divergence_bits_retained_vocabulary",
            ),
        )
    for record in length.to_dict(orient="records"):
        _add(
            "document_length",
            str(record["text_representation"]),
            record,
            (
                "target_to_source_mean_ratio",
                "target_to_source_median_ratio",
                "standardized_mean_difference_target_minus_source",
                "kolmogorov_smirnov_statistic",
            ),
        )
    for record in categorical.to_dict(orient="records"):
        _add(
            "composition",
            str(record["field"]),
            record,
            (
                "category_jaccard",
                "target_record_share_in_source_categories",
                "total_variation_distance",
                "jensen_shannon_divergence_bits",
                "target_minus_source_unreliable_prevalence",
            ),
        )
    if not topic.empty:
        _add(
            "topic_proxy",
            "full_text",
            topic.iloc[0].to_dict(),
            (
                "jensen_shannon_divergence_bits_mean_topic_weights",
                "total_variation_distance_mean_topic_weights",
                "most_target_enriched_topic_delta",
                "most_source_enriched_topic_delta",
            ),
        )
    _add(
        "domain_exclusion",
        "source_domain",
        exclusion.iloc[0].to_dict(),
        (
            "target_records_removed",
            "target_record_removal_share",
            "target_domains_removed",
        ),
    )
    return pd.DataFrame(rows)


def _markdown_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[object]],
) -> str:
    def _format(value: object) -> str:
        if value is None or (
            isinstance(value, (float, np.floating)) and math.isnan(float(value))
        ):
            return "NA"
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.3f}"
        return str(value).replace("|", r"\|")

    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *[
                "| " + " | ".join(_format(value) for value in row) + " |"
                for row in rows
            ],
        ]
    )


def render_degradation_markdown(degradation: pd.DataFrame) -> str:
    """Render compatible degradation estimates and explicit unavailable reasons."""

    required = {
        "transfer_experiment_id",
        "comparator_kind",
        "metric",
        "comparison_status",
        "unavailable_reason",
        "in_domain_comparator_value",
        "transfer_value",
        "absolute_delta_transfer_minus_in_domain",
        "relative_delta_transfer_minus_in_domain",
    }
    if not required <= set(degradation.columns):
        raise ValueError("Degradation table schema is incomplete.")
    lines = [
        "# In-domain versus cross-dataset transfer degradation",
        "",
        (
            "The primary comparator is a target-trained standard-split baseline "
            "with the same estimator family and text representation, recomputed "
            "on the exact post-domain-exclusion target records. The secondary "
            "source-domain comparator is the conventional source-test-to-transfer "
            "drop; it changes records and label construct and is descriptive only."
        ),
        "",
        (
            "Absolute delta is `transfer − in-domain`; negative values indicate "
            "lower transfer performance. Relative delta divides that quantity by "
            "the in-domain value. Undefined comparisons remain unavailable."
        ),
    ]
    for comparator_kind, heading in (
        ("target_in_domain_same_records", "Primary same-record comparator"),
        ("source_in_domain_standard_test", "Secondary source-test comparator"),
    ):
        subset = degradation[
            degradation["comparator_kind"].eq(comparator_kind)
        ]
        lines.extend(
            [
                "",
                f"## {heading}",
                "",
                _markdown_table(
                    (
                        "Transfer experiment",
                        "Metric",
                        "In-domain",
                        "Transfer",
                        "Absolute Δ",
                        "Relative Δ",
                        "Status / reason",
                    ),
                    [
                        (
                            row["transfer_experiment_id"],
                            row["metric"],
                            row["in_domain_comparator_value"],
                            row["transfer_value"],
                            row[
                                "absolute_delta_transfer_minus_in_domain"
                            ],
                            row[
                                "relative_delta_transfer_minus_in_domain"
                            ],
                            (
                                row["comparison_status"]
                                if row["comparison_status"] == "available"
                                else (
                                    f"unavailable: "
                                    f"{row['unavailable_reason']}"
                                )
                            ),
                        )
                        for _, row in subset.iterrows()
                    ],
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            (
                "- The primary comparison holds target evaluation records fixed "
                "but changes the training dataset and validation-selected "
                "hyperparameters."
            ),
            (
                "- The secondary comparison holds the model family and text "
                "representation fixed but changes evaluation records and label "
                "provenance."
            ),
            (
                "- Neither delta identifies a causal source of degradation; the "
                "separate shift tables describe candidate covariate differences."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def render_shift_markdown(
    *,
    direction: str,
    source_dataset: str,
    target_dataset: str,
    exclusion_summary: pd.DataFrame,
    vocabulary_summary: pd.DataFrame,
    length_shift: pd.DataFrame,
    categorical_summary: pd.DataFrame,
    topic_summary: pd.DataFrame,
) -> str:
    """Render a compact Markdown interpretation alongside machine-readable tables."""

    exclusion = exclusion_summary.iloc[0]
    length_by_representation = length_shift.set_index("text_representation")
    vocabulary_rows = []
    for _, row in vocabulary_summary.iterrows():
        if row["status"] != "ok":
            continue
        length = length_by_representation.loc[row["text_representation"]]
        vocabulary_rows.append(
            (
                row["text_representation"],
                row["active_type_jaccard"],
                row["target_token_coverage_by_source_active_types"],
                row["jensen_shannon_divergence_bits_retained_vocabulary"],
                length["kolmogorov_smirnov_statistic"],
                length["standardized_mean_difference_target_minus_source"],
            )
        )
    lines = [
        f"# Transfer shift diagnostics: {source_dataset} → {target_dataset}",
        "",
        (
            f"Direction key: `{direction}`. Metrics compare the source training "
            "cohort with the target test cohort after source-domain exclusion."
        ),
        "",
        (
            f"Domain exclusion removed {int(exclusion['target_records_removed'])} "
            f"of {int(exclusion['target_records_before_exclusion'])} target records "
            f"({float(exclusion['target_record_removal_share']):.1%})."
        ),
        "",
        "## Lexical and length shift",
        "",
        _markdown_table(
            (
                "Representation",
                "Type Jaccard",
                "Target token coverage",
                "Vocabulary JS (bits)",
                "Length KS",
                "Length SMD",
            ),
            vocabulary_rows,
        ),
        "",
        "## Label, text-unit, and source composition",
        "",
        _markdown_table(
            (
                "Field",
                "Category Jaccard",
                "Target record overlap",
                "Total variation",
                "JS (bits)",
            ),
            [
                (
                    row["field"],
                    row["category_jaccard"],
                    row["target_record_share_in_source_categories"],
                    row["total_variation_distance"],
                    row["jensen_shannon_divergence_bits"],
                )
                for _, row in categorical_summary.iterrows()
            ],
        ),
        "",
        "## Topic proxy",
        "",
    ]
    if topic_summary.empty or topic_summary.iloc[0]["status"] != "ok":
        lines.append("The pooled TF-IDF/NMF topic proxy could not be fitted.")
    else:
        topic = topic_summary.iloc[0]
        lines.extend(
            [
                (
                    f"The deterministic {int(topic['fitted_topics'])}-component "
                    "pooled TF-IDF/NMF proxy has a topic-mixture JS divergence of "
                    f"{float(topic['jensen_shannon_divergence_bits_mean_topic_weights']):.3f} "
                    "bits."
                ),
                "",
                (
                    f"Most target-enriched component: "
                    f"`{topic['most_target_enriched_topic']}` "
                    f"(Δ share {float(topic['most_target_enriched_topic_delta']):+.3f}); "
                    f"top terms: {topic['most_target_enriched_topic_terms']}."
                ),
                "",
                (
                    f"Most source-enriched component: "
                    f"`{topic['most_source_enriched_topic']}` "
                    f"(Δ share {float(topic['most_source_enriched_topic_delta']):+.3f}); "
                    f"top terms: {topic['most_source_enriched_topic_terms']}."
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            (
                "- These are descriptive, post-hoc shift diagnostics. They do not "
                "causally attribute transfer errors to any one dimension."
            ),
            (
                "- CoAID and FakeHealth labels have different provenance, so "
                "prevalence and performance differences combine construct shift "
                "with covariate shift."
            ),
            (
                "- TF-IDF/NMF components are a reproducible topic proxy, not "
                "validated narratives; they encode neither stance nor truth."
            ),
            (
                "- Vocabulary metrics use a capped pooled unigram space; detailed "
                "term tables expose retained feature frequencies."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def build_transfer_shift_artifacts(
    source_train: pd.DataFrame,
    target_test_before_exclusion: pd.DataFrame,
    target_test_after_exclusion: pd.DataFrame,
    *,
    source_dataset: str,
    target_dataset: str,
    exclusion_source_domains: Iterable[object],
    random_seed: int = 613,
    vocabulary_max_features: int = 20_000,
    vocabulary_minimum_term_count: int = 2,
    topic_requested_components: int = 10,
    topic_max_features: int = 5_000,
    topic_top_terms: int = 10,
) -> TransferShiftArtifacts:
    """Build all transfer-shift artifacts without writing to disk."""

    direction = f"{source_dataset}_to_{target_dataset}"
    required_columns = {
        "record_id",
        "harmonized_label",
        "text_unit",
        "publisher_id",
        "source_domain",
        "model_text",
        "title",
    }
    cohorts = (
        (SOURCE_COHORT, source_train),
        ("target_test_before_domain_exclusion", target_test_before_exclusion),
        (TARGET_COHORT, target_test_after_exclusion),
    )
    for cohort, frame in cohorts:
        missing = required_columns - set(frame.columns)
        if missing:
            raise ValueError(f"{cohort} is missing columns: {sorted(missing)}")
        if frame.empty:
            raise ValueError(f"{cohort} must contain at least one record.")

    exclusion_domains = {
        domain for domain in _texts(exclusion_source_domains) if domain.strip()
    }
    before_ids = set(target_test_before_exclusion["record_id"].astype(str))
    after_ids = set(target_test_after_exclusion["record_id"].astype(str))
    removed = target_test_before_exclusion[
        target_test_before_exclusion["record_id"]
        .astype(str)
        .isin(before_ids - after_ids)
    ]
    removed_domains = {
        domain for domain in _texts(removed["source_domain"]) if domain.strip()
    }
    before_domains = _categories(target_test_before_exclusion["source_domain"])
    after_domains = _categories(target_test_after_exclusion["source_domain"])
    exclusion_summary = pd.DataFrame(
        [
            {
                "direction": direction,
                "source_domain_reference_count": len(exclusion_domains),
                "target_records_before_exclusion": len(
                    target_test_before_exclusion
                ),
                "target_records_after_exclusion": len(
                    target_test_after_exclusion
                ),
                "target_records_removed": len(removed),
                "target_record_removal_share": (
                    len(removed) / len(target_test_before_exclusion)
                ),
                "target_domains_before_exclusion": int(
                    before_domains.nunique()
                ),
                "target_domains_after_exclusion": int(after_domains.nunique()),
                "target_domains_removed": len(removed_domains),
                "remaining_target_records_on_exclusion_domains": int(
                    after_domains.isin(exclusion_domains).sum()
                ),
            }
        ]
    )
    cohort_profiles = pd.DataFrame(
        [
            _cohort_profile(
                source_train,
                direction=direction,
                dataset=source_dataset,
                cohort=SOURCE_COHORT,
            ),
            _cohort_profile(
                target_test_before_exclusion,
                direction=direction,
                dataset=target_dataset,
                cohort="target_test_before_domain_exclusion",
            ),
            _cohort_profile(
                target_test_after_exclusion,
                direction=direction,
                dataset=target_dataset,
                cohort=TARGET_COHORT,
            ),
        ]
    )

    vocabulary_summaries: list[dict[str, Any]] = []
    vocabulary_terms: list[pd.DataFrame] = []
    length_profiles: list[pd.DataFrame] = []
    length_shifts: list[dict[str, Any]] = []
    for representation, text_column in (
        ("full_text", "model_text"),
        ("title_only", "title"),
    ):
        terms, vocabulary = vocabulary_shift_diagnostics(
            source_train[text_column],
            target_test_after_exclusion[text_column],
            max_features=vocabulary_max_features,
            minimum_term_count=vocabulary_minimum_term_count,
        )
        vocabulary_summaries.append(
            {
                "direction": direction,
                "text_representation": representation,
                **vocabulary,
            }
        )
        if not terms.empty:
            terms.insert(0, "text_representation", representation)
            terms.insert(0, "direction", direction)
            vocabulary_terms.append(terms)
        profiles, length = document_length_diagnostics(
            source_train[text_column],
            target_test_after_exclusion[text_column],
        )
        profiles.insert(0, "text_representation", representation)
        profiles.insert(0, "direction", direction)
        length_profiles.append(profiles)
        length_shifts.append(
            {
                "direction": direction,
                "text_representation": representation,
                **length,
            }
        )

    categorical_distributions: list[pd.DataFrame] = []
    categorical_summaries: list[dict[str, Any]] = []
    for field in (
        "harmonized_label",
        "text_unit",
        "publisher_id",
        "source_domain",
    ):
        distribution, summary = categorical_shift_diagnostics(
            source_train[field],
            target_test_after_exclusion[field],
            field=field,
        )
        distribution.insert(0, "direction", direction)
        categorical_distributions.append(distribution)
        categorical_summaries.append({"direction": direction, **summary})

    topic_terms, topic_distribution, topic_assignments, topic = (
        topic_proxy_diagnostics(
            source_train["model_text"],
            target_test_after_exclusion["model_text"],
            source_ids=source_train["record_id"],
            target_ids=target_test_after_exclusion["record_id"],
            random_seed=random_seed,
            requested_topics=topic_requested_components,
            max_features=topic_max_features,
            top_terms_per_topic=topic_top_terms,
        )
    )
    for frame in (topic_terms, topic_distribution, topic_assignments):
        if not frame.empty:
            frame.insert(0, "direction", direction)

    vocabulary_summary = pd.DataFrame(vocabulary_summaries)
    length_shift = pd.DataFrame(length_shifts)
    categorical_summary = pd.DataFrame(categorical_summaries)
    topic_summary = pd.DataFrame([{"direction": direction, **topic}])
    tables = {
        "cohort_profiles": cohort_profiles,
        "domain_exclusion_summary": exclusion_summary,
        "vocabulary_summary": vocabulary_summary,
        "vocabulary_shift_terms": (
            pd.concat(vocabulary_terms, ignore_index=True)
            if vocabulary_terms
            else pd.DataFrame()
        ),
        "document_length_profiles": pd.concat(
            length_profiles, ignore_index=True
        ),
        "document_length_shift": length_shift,
        "categorical_distributions": pd.concat(
            categorical_distributions, ignore_index=True
        ),
        "categorical_shift_summary": categorical_summary,
        "topic_proxy_top_terms": topic_terms,
        "topic_proxy_distributions": topic_distribution,
        "topic_proxy_assignments": topic_assignments,
        "topic_proxy_summary": topic_summary,
    }
    tables["shift_summary"] = _long_summary(
        direction,
        exclusion=exclusion_summary,
        vocabulary=vocabulary_summary,
        length=length_shift,
        categorical=categorical_summary,
        topic=topic_summary,
    )
    metadata = {
        "direction": direction,
        "source_dataset": source_dataset,
        "target_dataset": target_dataset,
        "comparison_cohorts": {
            "source": SOURCE_COHORT,
            "target": TARGET_COHORT,
        },
        "parameters": {
            "random_seed": random_seed,
            "vocabulary_max_features": vocabulary_max_features,
            "vocabulary_minimum_term_count": (
                vocabulary_minimum_term_count
            ),
            "topic_method": "pooled_tfidf_nmf",
            "topic_requested_components": topic_requested_components,
            "topic_max_features": topic_max_features,
            "topic_top_terms": topic_top_terms,
        },
        "interpretation": {
            "purpose": (
                "Post-hoc descriptive construct/covariate-shift diagnostics for "
                "interpreting cross-dataset transfer."
            ),
            "causal_attribution": False,
            "topic_components_are_validated_narratives": False,
            "labels_assumed_equivalent": False,
        },
    }
    markdown = render_shift_markdown(
        direction=direction,
        source_dataset=source_dataset,
        target_dataset=target_dataset,
        exclusion_summary=exclusion_summary,
        vocabulary_summary=vocabulary_summary,
        length_shift=length_shift,
        categorical_summary=categorical_summary,
        topic_summary=topic_summary,
    )
    return TransferShiftArtifacts(metadata, tables, markdown)
