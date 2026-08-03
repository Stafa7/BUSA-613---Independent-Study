from health_misinfo.models.baselines import baseline_candidates, tfidf_parameter_grid


def _config():
    return {
        "random_seed": 613,
        "models": {
            "tfidf": {
                "max_features": 20000,
                "max_features_grid": [10000, 20000],
                "ngram_range": [1, 2],
                "min_df": 2,
                "min_df_grid": [5, 2],
                "sublinear_tf": False,
                "sublinear_tf_grid": [False, True],
            },
            "logistic_regression": {
                "max_iter": 1000,
                "class_weight": "balanced",
                "c_grid": [0.1, 1.0, 10.0],
            },
            "linear_svm": {
                "class_weight": "balanced",
                "c_grid": [0.1, 1.0, 10.0],
            },
        },
    }


def test_tfidf_grid_is_frozen_and_deterministic():
    grid = tfidf_parameter_grid(_config())

    assert len(grid) == 8
    assert grid[0] == {
        "tfidf_max_features": 10000,
        "tfidf_ngram_min": 1,
        "tfidf_ngram_max": 2,
        "tfidf_min_df": 5,
        "tfidf_sublinear_tf": False,
    }
    assert len({tuple(sorted(candidate.items())) for candidate in grid}) == len(grid)


def test_each_linear_model_searches_representation_and_regularization():
    candidates = baseline_candidates(_config())

    assert len(candidates["logistic_regression"]) == 24
    assert len(candidates["linear_svm"]) == 24
    assert all(
        candidate.estimator.named_steps["clf"].random_state == 613
        for candidate in candidates["linear_svm"]
    )
    assert candidates["logistic_regression"][0].hyperparameters["C"] == 0.1
    assert candidates["logistic_regression"][-1].hyperparameters["C"] == 10.0
