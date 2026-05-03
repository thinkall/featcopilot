"""Tests for the benchmark non-numeric encoding helper."""

from __future__ import annotations

import numpy as np
import pandas as pd

from benchmarks.simple_models.run_simple_models_benchmark import (
    _label_encode_non_numeric,
    preprocess_data,
    preprocess_target,
    preprocess_X_train_test,
    sanitize_columns,
)


def test_encoder_fits_on_train_only_and_handles_unseen_categories():
    """Unseen test categories must collapse to a single sentinel code (no leakage)."""
    X_train = pd.DataFrame({"num": [1.0, 2.0, 3.0], "cat": ["a", "b", "a"]})
    X_test = pd.DataFrame({"num": [4.0, 5.0], "cat": ["a", "z"]})  # "z" never appears in train

    X_train_enc, X_test_enc = _label_encode_non_numeric(X_train, X_test)

    # All columns numeric.
    assert X_train_enc["cat"].dtype.kind in "iu"
    assert X_test_enc["cat"].dtype.kind in "iu"
    # Train vocab is {"a", "b"} → codes 0, 1; unknown bucket = 2.
    assert set(X_train_enc["cat"].unique()) <= {0, 1}
    seen_code = X_train_enc["cat"].iloc[0]  # code for "a"
    assert X_test_enc["cat"].iloc[0] == seen_code
    assert X_test_enc["cat"].iloc[1] == 2  # unknown bucket = len(train_classes)
    # Numeric columns pass through untouched.
    assert (X_train_enc["num"].values == X_train["num"].values).all()


def test_encoder_replaces_nan_before_string_cast():
    """NaN must become the ``missing`` sentinel, not the literal string ``"nan"``.

    If ``astype(str)`` runs before ``fillna``, NaNs become ``"nan"`` and live in
    a separate code from real ``"missing"`` strings — a subtle bug. This test
    pins the correct ordering.
    """
    X_train = pd.DataFrame({"cat": ["x", None, "missing"]})
    X_test = pd.DataFrame({"cat": [None, "x"]})

    X_train_enc, X_test_enc = _label_encode_non_numeric(X_train, X_test)

    # Train classes = {"missing", "x"}; NaN should map to the same code as the
    # literal "missing" string.
    nan_code = X_train_enc["cat"].iloc[1]
    missing_code = X_train_enc["cat"].iloc[2]
    assert nan_code == missing_code
    # Test NaN should map to the same train "missing" code, NOT the unknown bucket.
    assert X_test_enc["cat"].iloc[0] == missing_code


def test_encoder_passes_through_when_all_numeric():
    """No non-numeric columns → frames returned unchanged (modulo copy)."""
    X_train = pd.DataFrame({"a": [1.0, 2.0], "b": [3, 4]})
    X_test = pd.DataFrame({"a": [5.0, 6.0], "b": [7, 8]})

    X_train_enc, X_test_enc = _label_encode_non_numeric(X_train, X_test)

    pd.testing.assert_frame_equal(X_train_enc, X_train)
    pd.testing.assert_frame_equal(X_test_enc, X_test)
    # Defensive copy — mutating the result must not affect the input.
    X_train_enc.iloc[0, 0] = 999.0
    assert X_train.iloc[0, 0] != 999.0


def test_encoder_handles_categorical_dtype_without_raising():
    """Pandas ``Categorical`` dtype must not crash even when "missing" isn't a category.

    Previously the code did ``series.fillna("missing")`` directly on Categoricals,
    which raises ``ValueError: fill value must be in categories``. The
    ``_safe_fillna_string`` helper casts to object first to avoid this.
    """
    X_train = pd.DataFrame({"cat": pd.Categorical(["a", "b", None, "a"], categories=["a", "b"])})
    X_test = pd.DataFrame({"cat": pd.Categorical(["b", None, "z"], categories=["a", "b", "z"])})

    # Must not raise.
    X_train_enc, X_test_enc = _label_encode_non_numeric(X_train, X_test)

    # NaN cells should map to the same code as the literal "missing" sentinel.
    assert X_train_enc["cat"].dtype.kind in "iu"
    assert X_test_enc["cat"].dtype.kind in "iu"
    # Test row with "z" (unseen in train) → unknown bucket. Length 3 train classes
    # ("a", "b", "missing") → unknown_code = 3.
    assert X_test_enc["cat"].iloc[2] == 3


def test_encoder_handles_categorical_dtype_with_missing_category_present():
    """If "missing" already exists as a category, encoding must still succeed."""
    X_train = pd.DataFrame({"cat": pd.Categorical(["a", "missing", None], categories=["a", "missing"])})
    X_test = pd.DataFrame({"cat": pd.Categorical(["a", None], categories=["a"])})

    X_train_enc, X_test_enc = _label_encode_non_numeric(X_train, X_test)

    # All NaN and explicit "missing" cells in train share one code.
    train_classes = set(X_train_enc["cat"].unique())
    assert len(train_classes) == 2  # {"a", "missing"} → 2 codes


def test_encoder_handles_datetime_columns_without_raising():
    """Datetime columns must not crash on the string sentinel.

    ``pd.Series.fillna("missing")`` raises on ``datetime64`` dtype because
    the string is not type-compatible. ``select_dtypes(exclude=[np.number])``
    *does* include datetime64 columns (timedelta64 is treated as numeric and
    excluded, which is fine — it stays untouched). The ``_safe_fillna_string``
    helper casts to object first so the encoder works on datetime dtypes.
    """
    X_train = pd.DataFrame(
        {
            "ts": pd.to_datetime(["2024-01-01", "2024-01-02", pd.NaT, "2024-01-04"]),
        }
    )
    X_test = pd.DataFrame(
        {
            "ts": pd.to_datetime(["2024-01-02", pd.NaT, "2024-02-01"]),
        }
    )

    # Must not raise on datetime dtype.
    X_train_enc, X_test_enc = _label_encode_non_numeric(X_train, X_test)

    # Encoded as int.
    assert X_train_enc["ts"].dtype.kind in "iu"
    assert X_test_enc["ts"].dtype.kind in "iu"

    # Unseen test value ("2024-02-01") → unknown bucket.
    train_classes = set(X_train_enc["ts"].unique())
    assert X_test_enc["ts"].iloc[2] not in train_classes


def test_encoder_preserves_numeric_columns_alongside_categorical():
    """Mixed-dtype frames: only non-numeric columns are encoded."""
    rng = np.random.default_rng(0)
    X_train = pd.DataFrame(
        {
            "num1": rng.standard_normal(10),
            "cat1": ["a", "b"] * 5,
            "num2": np.arange(10),
        }
    )
    X_test = pd.DataFrame(
        {
            "num1": rng.standard_normal(4),
            "cat1": ["a", "b", "a", "b"],
            "num2": np.arange(4),
        }
    )

    X_train_enc, X_test_enc = _label_encode_non_numeric(X_train, X_test)

    assert (X_train_enc["num1"].values == X_train["num1"].values).all()
    assert (X_train_enc["num2"].values == X_train["num2"].values).all()
    assert X_train_enc["cat1"].dtype.kind in "iu"
    assert X_test_enc["cat1"].dtype.kind in "iu"


def test_preprocess_data_handles_categorical_dtype():
    """``preprocess_data`` must encode pandas Categorical columns without raising.

    Pre-fix this would raise ``ValueError: fill value must be in categories`` on
    the ``fillna("missing")`` step because "missing" wasn't a category.
    """
    X = pd.DataFrame(
        {
            "num": [1.0, 2.0, np.nan],
            "cat": pd.Categorical(["a", "b", None], categories=["a", "b"]),
        }
    )
    y = pd.Series([0, 1, 0])

    # Must not raise.
    X_processed, y_processed = preprocess_data(X, y, "classification")

    assert X_processed["num"].notna().all()  # NaN filled with median
    assert X_processed["cat"].dtype.kind in "iu"  # categorical → integer codes
    assert len(y_processed) == 3


# ---------------------------------------------------------------------------
# Fold-local preprocessing leakage regression tests
# ---------------------------------------------------------------------------


def test_preprocess_X_train_test_uses_train_only_medians():
    """Numeric NaN imputation must use the TRAIN median, not the global one.

    Prior to the fix, ``preprocess_data`` was applied once to the full dataset
    before the CV split, so train rows were imputed using statistics computed
    over both train AND test — biasing the paired Wilcoxon p-values used in
    the published benchmark methodology.
    """
    # Train = [1, 2, 3, 4, 5] -> median = 3.0; Test = [None, None, 100] would
    # shift the global median to 4.0 if the (bug-prone) full-dataset
    # imputation were used. We assert imputed values come from the train
    # median (3.0) not the global one.
    X_train = pd.DataFrame({"num": [1.0, 2.0, 3.0, 4.0, 5.0]})
    X_test = pd.DataFrame({"num": [np.nan, np.nan, 100.0]})

    X_train_proc, X_test_proc = preprocess_X_train_test(X_train, X_test)

    assert X_train_proc["num"].notna().all()
    assert X_test_proc["num"].iloc[0] == 3.0
    assert X_test_proc["num"].iloc[1] == 3.0
    # Global median (with the imbalanced [1..5,nan,nan,100]) would be 3.5; the
    # difference is the leak signal we're guarding against.


def test_preprocess_X_train_test_uses_train_only_categories():
    """Non-numeric encoder vocab must be fit on TRAIN ONLY (no leakage)."""
    # "z" only ever appears in test; train vocab = {"a","b"}. With a leakage-prone
    # implementation, "z" would receive a known code. We assert it lands in the
    # unknown bucket = len(train_classes) instead.
    X_train = pd.DataFrame({"cat": ["a", "b", "a", "b"]})
    X_test = pd.DataFrame({"cat": ["a", "z", "b"]})

    X_train_proc, X_test_proc = preprocess_X_train_test(X_train, X_test)

    train_unique = set(X_train_proc["cat"].unique())
    # Vocab = {"a","b"} → codes {0,1}; unknown bucket = 2.
    assert train_unique <= {0, 1}
    assert X_test_proc["cat"].iloc[1] == 2


def test_preprocess_target_independence_from_X():
    """``preprocess_target`` must not depend on X (target alone is fold-safe)."""
    y = pd.Series(["a", "b", "a", "c"])
    encoded = preprocess_target(y, "classification")
    assert encoded.shape == (4,)
    assert set(encoded.tolist()) == {0, 1, 2}

    # Regression should pass through as float.
    y_reg = pd.Series([1, 2, 3])
    out = preprocess_target(y_reg, "regression")
    assert out.dtype == np.float64


def test_sanitize_columns_is_pure_rename():
    """``sanitize_columns`` is a pure rename — values must pass through unchanged."""
    X = pd.DataFrame({"col with space": [1.0, 2.0], "col/with/slash": ["a", "b"]})
    renamed = sanitize_columns(X)
    # Sanitized column names must not contain whitespace or slashes.
    assert all(" " not in c and "/" not in c for c in renamed.columns)
    # Values are untouched.
    assert (renamed.iloc[:, 0].values == X.iloc[:, 0].values).all()
    assert (renamed.iloc[:, 1].values == X.iloc[:, 1].values).all()


def test_generate_report_header_collapses_uniform_seeds(tmp_path):
    """Single-value n_folds/n_seeds across results render as a single number, no note."""
    from benchmarks.simple_models.run_simple_models_benchmark import generate_report

    results = [
        {
            "dataset": "d1",
            "task": "classification",
            "source": "real_world",
            "n_samples": 100,
            "n_features_original": 5,
            "n_folds": 5,
            "n_seeds": 3,
            "with_llm": False,
            "baseline_best_score": 0.9,
            "baseline_std": 0.01,
            "tabular_best_score": 0.91,
            "tabular_std": 0.01,
            "tabular_improvement_pct": 1.1,
            "p_value": 0.04,
            "significant": True,
            "n_features_tabular": 6,
            "fe_time_tabular": 0.5,
            "engines_used": ["tabular"],
            "baseline_fold_scores": [0.9] * 5,
            "tabular_fold_scores": [0.91] * 5,
        },
        {
            "dataset": "d2",
            "task": "classification",
            "source": "real_world",
            "n_samples": 200,
            "n_features_original": 8,
            "n_folds": 5,
            "n_seeds": 3,
            "with_llm": False,
            "baseline_best_score": 0.7,
            "baseline_std": 0.02,
            "tabular_best_score": 0.7,
            "tabular_std": 0.02,
            "tabular_improvement_pct": 0.0,
            "p_value": 1.0,
            "significant": False,
            "n_features_tabular": 8,
            "fe_time_tabular": 0.4,
            "engines_used": ["tabular"],
            "baseline_fold_scores": [0.7] * 5,
            "tabular_fold_scores": [0.7] * 5,
        },
    ]

    out_dir = tmp_path
    out_dir.mkdir(exist_ok=True)
    generate_report(results, with_llm=False, output_path=out_dir)
    text = (out_dir / "SIMPLE_MODELS_BENCHMARK.md").read_text(encoding="utf-8")

    assert "**Cross-Validation:** 5-fold CV × 3 seed(s)" in text
    # No note when all values agree.
    assert "n_seeds`` varies across datasets" not in text


def test_generate_report_header_surfaces_mixed_seeds(tmp_path):
    """Mixed n_seeds across results render as a range and emit the time-series note."""
    from benchmarks.simple_models.run_simple_models_benchmark import generate_report

    base = {
        "dataset": "d1",
        "task": "classification",
        "source": "real_world",
        "n_samples": 100,
        "n_features_original": 5,
        "n_folds": 5,
        "n_seeds": 3,
        "with_llm": False,
        "baseline_best_score": 0.9,
        "baseline_std": 0.01,
        "tabular_best_score": 0.91,
        "tabular_std": 0.01,
        "tabular_improvement_pct": 1.1,
        "p_value": 0.04,
        "significant": True,
        "n_features_tabular": 6,
        "fe_time_tabular": 0.5,
        "engines_used": ["tabular"],
        "baseline_fold_scores": [0.9] * 5,
        "tabular_fold_scores": [0.91] * 5,
    }
    ts = dict(base)
    ts["dataset"] = "d_ts"
    ts["task"] = "timeseries_regression"
    ts["source"] = "real_world"
    ts["n_seeds"] = 1  # collapsed by TimeSeriesSplit

    out_dir = tmp_path
    out_dir.mkdir(exist_ok=True)
    generate_report([base, ts], with_llm=False, output_path=out_dir)
    text = (out_dir / "SIMPLE_MODELS_BENCHMARK.md").read_text(encoding="utf-8")

    assert "**Cross-Validation:** 5-fold CV × 1–3 seed(s)" in text
    assert "n_seeds`` varies across datasets" in text


# ---------------------------------------------------------------------------
# CLI argument validation
# ---------------------------------------------------------------------------


def test_main_rejects_invalid_n_folds(monkeypatch, capsys):
    """``main`` must fail fast when --n-folds < 2 (CV requires 2+ folds)."""
    import pytest

    from benchmarks.simple_models import run_simple_models_benchmark as bench

    monkeypatch.setattr(
        "sys.argv",
        ["run_simple_models_benchmark.py", "--datasets", "d1", "--n-folds", "1"],
    )
    with pytest.raises(SystemExit) as exc_info:
        bench.main()
    # argparse parser.error exits with code 2.
    assert exc_info.value.code == 2
    err = capsys.readouterr().err
    assert "--n-folds" in err and ">= 2" in err


def test_main_rejects_invalid_n_seeds(monkeypatch, capsys):
    """``main`` must fail fast when --n-seeds < 1 (would silently skip every dataset)."""
    import pytest

    from benchmarks.simple_models import run_simple_models_benchmark as bench

    monkeypatch.setattr(
        "sys.argv",
        ["run_simple_models_benchmark.py", "--datasets", "d1", "--n-seeds", "0"],
    )
    with pytest.raises(SystemExit) as exc_info:
        bench.main()
    assert exc_info.value.code == 2
    err = capsys.readouterr().err
    assert "--n-seeds" in err and ">= 1" in err


# ---------------------------------------------------------------------------
# Report task-bucketing regression tests
# ---------------------------------------------------------------------------


def test_generate_report_buckets_text_regression_into_regression(tmp_path):
    """Tasks containing 'regression' (e.g. 'text_regression', 'timeseries_regression')
    must land in the Regression table, not the catch-all Other table — and pick up
    the R² metric label rather than 'Score'."""
    from benchmarks.simple_models.run_simple_models_benchmark import generate_report

    base_row = {
        "n_samples": 100,
        "n_features_original": 5,
        "n_folds": 5,
        "n_seeds": 1,
        "with_llm": False,
        "baseline_best_score": 0.7,
        "baseline_std": 0.01,
        "tabular_best_score": 0.71,
        "tabular_std": 0.01,
        "tabular_improvement_pct": 1.4,
        "p_value": 0.04,
        "significant": True,
        "n_features_tabular": 6,
        "fe_time_tabular": 0.5,
        "engines_used": ["tabular"],
        "baseline_fold_scores": [0.7] * 5,
        "tabular_fold_scores": [0.71] * 5,
    }
    results = [
        {**base_row, "dataset": "ts_reg", "task": "timeseries_regression", "source": "real_world"},
        {**base_row, "dataset": "text_reg", "task": "text_regression", "source": "real_world"},
    ]

    out_dir = tmp_path
    out_dir.mkdir(exist_ok=True)
    generate_report(results, with_llm=False, output_path=out_dir)
    text = (out_dir / "SIMPLE_MODELS_BENCHMARK.md").read_text(encoding="utf-8")

    # Both rows must appear under the Regression table (R² header), not Other.
    assert "Real-World Regression" in text
    assert "ts_reg" in text and "text_reg" in text
    # If they had been mis-bucketed into "Other", the section title would appear.
    assert "Real-World Other" not in text


def test_generate_report_buckets_text_classification_into_classification(tmp_path):
    """Regression-guard the classification substring matching too — make sure
    the both-buckets symmetry isn't accidentally broken by the regression fix."""
    from benchmarks.simple_models.run_simple_models_benchmark import generate_report

    base_row = {
        "n_samples": 100,
        "n_features_original": 5,
        "n_folds": 5,
        "n_seeds": 1,
        "with_llm": False,
        "baseline_best_score": 0.85,
        "baseline_std": 0.01,
        "tabular_best_score": 0.86,
        "tabular_std": 0.01,
        "tabular_improvement_pct": 1.2,
        "p_value": 0.04,
        "significant": True,
        "n_features_tabular": 6,
        "fe_time_tabular": 0.5,
        "engines_used": ["tabular"],
        "baseline_fold_scores": [0.85] * 5,
        "tabular_fold_scores": [0.86] * 5,
    }
    results = [
        {**base_row, "dataset": "text_clf", "task": "text_classification", "source": "real_world"},
    ]

    out_dir = tmp_path
    out_dir.mkdir(exist_ok=True)
    generate_report(results, with_llm=False, output_path=out_dir)
    text = (out_dir / "SIMPLE_MODELS_BENCHMARK.md").read_text(encoding="utf-8")

    assert "Real-World Classification" in text
    assert "text_clf" in text
    assert "Real-World Other" not in text


# ---------------------------------------------------------------------------
# Round-22 fold-local preprocessing edge cases
# ---------------------------------------------------------------------------


def test_preprocess_X_train_test_handles_all_missing_train_column():
    """A numeric column that is entirely NaN in the train fold has a NaN
    median, so naive ``fillna(train_median)`` would be a no-op and leave
    NaNs for the model to choke on. The function must fall back to a
    fixed sentinel (0.0) and emit no NaNs."""
    X_train = pd.DataFrame({"all_nan": [np.nan, np.nan, np.nan, np.nan], "ok": [1.0, 2.0, 3.0, 4.0]})
    X_test = pd.DataFrame({"all_nan": [np.nan, 5.0, np.nan], "ok": [np.nan, 5.0, 6.0]})

    X_train_proc, X_test_proc = preprocess_X_train_test(X_train, X_test)

    assert X_train_proc["all_nan"].notna().all()
    assert X_test_proc["all_nan"].notna().all()
    # The all-missing train column collapses to the 0.0 sentinel.
    assert (X_train_proc["all_nan"] == 0.0).all()
    # Test rows without a value also fill with 0.0; the lone 5.0 survives.
    assert X_test_proc["all_nan"].iloc[0] == 0.0
    assert X_test_proc["all_nan"].iloc[1] == 5.0
    assert X_test_proc["all_nan"].iloc[2] == 0.0


def test_preprocess_target_handles_pandas_string_dtype():
    """Pandas ``StringDtype`` (``"string"``) targets are non-numeric
    extension dtypes, but the legacy implementation only checked for
    ``object``/``category`` and would fall through, returning a
    non-numeric ndarray that breaks sklearn estimators downstream."""
    y = pd.Series(["a", "b", "a", "c"], dtype="string")
    encoded = preprocess_target(y, "classification")
    assert encoded.shape == (4,)
    # Must be numeric (label-encoded), not still strings.
    assert np.issubdtype(encoded.dtype, np.integer) or np.issubdtype(encoded.dtype, np.floating)
    assert set(encoded.tolist()) == {0, 1, 2}


def test_preprocess_target_passes_through_numeric_classification():
    """Numeric classification targets (e.g. {0, 1}) must NOT be label-encoded."""
    y = pd.Series([0, 1, 0, 1, 1])
    encoded = preprocess_target(y, "classification")
    # Identity: numeric classification targets pass through unchanged.
    assert (encoded == y.values).all()


def test_preprocess_target_passes_through_bool_classification():
    """Bool classification targets must be treated as numeric (no encoding)."""
    y = pd.Series([True, False, True, False])
    encoded = preprocess_target(y, "classification")
    # Bool is a numeric dtype in pandas; passes through.
    assert encoded.shape == (4,)
    # Values comparable to original after a numeric cast.
    assert encoded.tolist() == [True, False, True, False] or encoded.tolist() == [1, 0, 1, 0]


def test_main_does_not_mutate_global_warnings_filter(monkeypatch):
    """Calling ``main()`` programmatically (e.g. from tests) must NOT
    silently suppress warnings for the rest of the process. The CLI-only
    ``filterwarnings("ignore")`` lives under ``if __name__ == "__main__"``."""
    import warnings as warnings_mod

    from benchmarks.simple_models import run_simple_models_benchmark as bench

    # Snapshot the warnings filter before invoking main().
    before = list(warnings_mod.filters)

    monkeypatch.setattr(
        "sys.argv",
        ["run_simple_models_benchmark.py", "--datasets", "d1", "--n-folds", "1"],
    )
    # main() will exit via parser.error (n_folds < 2); we only care that
    # it doesn't mutate the global filter on the way out.
    try:
        bench.main()
    except SystemExit:
        pass

    after = list(warnings_mod.filters)
    assert before == after, "main() must not mutate the global warnings filter"


# =============================================================================
# Round 23: source-inference helpers (_infer_source / _resolve_source)
# =============================================================================


def test_infer_source_synthetic_dataset_always_synthetic():
    """If the registry tags a dataset as synthetic, source is always synthetic
    regardless of loader_name."""
    from benchmarks.simple_models.run_simple_models_benchmark import _infer_source

    # ``titanic`` is registered as synthetic.
    assert _infer_source("titanic", "Titanic (Kaggle-style)") == "synthetic"
    assert _infer_source("titanic", "Titanic (Real)") == "synthetic"
    assert _infer_source("titanic", None) == "synthetic"


def test_infer_source_real_world_no_marker_returns_real_world():
    """Real-world dataset + non-synthetic loader_name → real_world."""
    from benchmarks.simple_models.run_simple_models_benchmark import _infer_source

    # ``fake_news`` is registered as real_world.
    assert _infer_source("fake_news", "Fake News (HuggingFace)") == "real_world"
    assert _infer_source("fake_news", None) == "real_world"


def test_infer_source_real_world_with_synthetic_fallback_marker():
    """Real-world dataset whose loader fell back to synthetic data must be
    downgraded to ``synthetic`` so the report split stays accurate."""
    from benchmarks.simple_models.run_simple_models_benchmark import _infer_source

    # Simulate a future Kaggle/HF loader for a real-world dataset that
    # falls back to synthetic data on fetch failure.
    assert _infer_source("fake_news", "Fake News (synthetic)") == "synthetic"
    assert _infer_source("fake_news", "Fake News (HF-style)") == "synthetic"
    # Case-insensitive marker matching.
    assert _infer_source("fake_news", "Fake News (SYNTHETIC)") == "synthetic"
    assert _infer_source("fake_news", "Fake News (Kaggle-Style)") == "synthetic"


def test_resolve_source_uses_explicit_field_when_present():
    """New-format results have ``source`` set; ``_resolve_source`` returns it verbatim."""
    from benchmarks.simple_models.run_simple_models_benchmark import _resolve_source

    assert _resolve_source({"source": "real_world", "dataset": "anything"}) == "real_world"
    assert _resolve_source({"source": "synthetic", "dataset": "anything"}) == "synthetic"
    # Even a value the registry would override is preserved (caller chose it).
    assert _resolve_source({"source": "real_world", "dataset": "titanic"}) == "real_world"


def test_resolve_source_falls_back_to_registry_for_legacy_results():
    """Legacy cached results predating the ``source`` field must be bucketed via
    ``is_real_world(dataset)``, not blindly treated as synthetic."""
    from benchmarks.simple_models.run_simple_models_benchmark import _resolve_source

    # ``fake_news`` is registered as real_world; legacy cached row should resolve to real_world.
    assert _resolve_source({"dataset": "fake_news"}) == "real_world"
    # ``titanic`` is registered as synthetic.
    assert _resolve_source({"dataset": "titanic"}) == "synthetic"
    # Unknown dataset → safe default ``synthetic``.
    assert _resolve_source({"dataset": "totally_unknown_dataset"}) == "synthetic"
    # Missing ``dataset`` field → safe default ``synthetic``.
    assert _resolve_source({}) == "synthetic"


def test_generate_report_buckets_legacy_results_via_registry(tmp_path):
    """``generate_report`` on legacy results (no ``source`` field) must bucket
    them via the registry instead of lumping all rows into ``synthetic``.

    Regression test for Copilot review on commit 61f0e9c: previously
    ``r.get("source") != "real_world"`` treated every legacy row as synthetic.
    """
    from benchmarks.simple_models.run_simple_models_benchmark import generate_report

    base_row = {
        "task": "classification",
        "n_samples": 100,
        "n_features_original": 5,
        "n_folds": 5,
        "n_seeds": 1,
        "with_llm": False,
        "baseline_best_score": 0.85,
        "baseline_std": 0.02,
        "tabular_best_score": 0.86,
        "tabular_std": 0.02,
        "tabular_improvement_pct": 1.0,
        "p_value": 0.04,
        "significant": True,
        "n_features_tabular": 8,
        "fe_time_tabular": 0.5,
        "engines_used": ["tabular"],
        "baseline_fold_scores": [0.85] * 5,
        "tabular_fold_scores": [0.86] * 5,
    }
    # Legacy results: NO ``source`` field at all.
    legacy_real = {**base_row, "dataset": "fake_news"}  # registry: real_world
    legacy_synth = {**base_row, "dataset": "titanic"}  # registry: synthetic

    out_dir = tmp_path / "report_dir"
    out_dir.mkdir()
    generate_report([legacy_real, legacy_synth], with_llm=False, output_path=out_dir)
    body = (out_dir / "SIMPLE_MODELS_BENCHMARK.md").read_text(encoding="utf-8")

    # Both real-world and synthetic sections must be populated; if legacy
    # results were all bucketed as synthetic the real-world section would
    # be empty / show 0 datasets.
    assert "fake_news" in body, "legacy real_world dataset must surface in report"
    assert "titanic" in body, "legacy synthetic dataset must surface in report"


# =============================================================================
# save_cache: numpy.bool_ JSON serialization (regression for full-benchmark
# crash where ``significant = p_value < 0.05`` is np.bool_ and json.dump fails)
# =============================================================================


def test_save_cache_handles_numpy_bool_in_significant_field(tmp_path):
    """``significant`` is ``p_value < 0.05`` where ``p_value`` is a numpy
    float, so the comparison yields ``numpy.bool_``. The default ``json``
    encoder treats ``numpy.bool_`` as non-serializable on NumPy <2 because
    it isn't a Python ``bool`` subclass for ``json``'s purposes.

    Regression test for a real crash observed during a full ``--all`` run:
    after benchmarking 63 datasets for 3+ hours, ``save_cache`` raised
    ``TypeError: Object of type bool is not JSON serializable`` and the
    in-memory results were lost. Cast to native ``bool`` explicitly.
    """
    import json as json_mod

    import numpy as np

    from benchmarks.simple_models.run_simple_models_benchmark import save_cache

    results = [
        {
            "dataset": "synth",
            "task": "classification",
            "source": "synthetic",
            "p_value": np.float64(0.03),
            "significant": np.bool_(True),
            "tabular_improvement_pct": np.float64(2.5),
            # Nested dict with numpy.bool_ should also work.
            "metadata": {"converged": np.bool_(False), "lr": np.float64(0.01)},
        }
    ]
    save_cache(results, tmp_path, with_llm=False)
    cache_file = tmp_path / "SIMPLE_MODELS_CACHE.json"
    loaded = json_mod.loads(cache_file.read_text(encoding="utf-8"))
    assert loaded[0]["significant"] is True
    assert isinstance(loaded[0]["significant"], bool)
    assert loaded[0]["metadata"]["converged"] is False
    assert isinstance(loaded[0]["metadata"]["converged"], bool)


def test_main_generates_report_even_when_save_cache_would_fail(monkeypatch, tmp_path):
    """If ``save_cache`` raises (e.g. a future numpy dtype the converter
    doesn't yet handle), the in-memory results should still produce a
    written report so the 3-hour run isn't wasted. Verify by patching
    ``save_cache`` to always raise."""
    import sys

    from benchmarks.simple_models import run_simple_models_benchmark as bench

    sample = {
        "dataset": "titanic",
        "task": "classification",
        "source": "synthetic",
        "loader_name": "Titanic (Kaggle-style)",
        "n_samples": 100,
        "n_features_original": 5,
        "n_folds": 5,
        "n_seeds": 1,
        "with_llm": False,
        "baseline_best_score": 0.8,
        "baseline_std": 0.01,
        "tabular_best_score": 0.81,
        "tabular_std": 0.01,
        "tabular_improvement_pct": 1.25,
        "p_value": 0.04,
        "significant": True,
        "n_features_tabular": 6,
        "fe_time_tabular": 0.1,
        "engines_used": ["tabular"],
        "baseline_fold_scores": [0.8] * 5,
        "tabular_fold_scores": [0.81] * 5,
    }
    monkeypatch.setattr(bench, "run_single_benchmark", lambda *a, **kw: sample)

    def boom(*a, **kw):
        raise TypeError("simulated serialization failure")

    monkeypatch.setattr(bench, "save_cache", boom)
    monkeypatch.setattr(
        sys,
        "argv",
        ["bench", "--datasets", "titanic", "--output", str(tmp_path)],
    )

    # main() should propagate the save_cache exception, but only AFTER
    # generate_report has already written the markdown.
    try:
        bench.main()
    except TypeError as e:
        assert "simulated serialization failure" in str(e)
    else:
        # If save_cache wasn't even called (e.g. no_cache=True path), the
        # report should still exist; that's acceptable too.
        pass

    report = tmp_path / "SIMPLE_MODELS_BENCHMARK.md"
    assert report.exists(), "report must be written even if save_cache raises"
    body = report.read_text(encoding="utf-8")
    assert "titanic" in body


def test_save_cache_preserves_int_fidelity_for_np_integer(tmp_path):
    """``np.integer`` values must serialize as JSON ints, not floats.

    Regression test for Copilot review on commit e9be75f: the previous
    converter used ``float(v)`` for both ``np.floating`` and
    ``np.integer``, turning ``n_samples=5000`` into ``5000.0`` in the
    cache and silently losing precision past ``2**53`` for very large
    integer counts. Splitting ``np.integer`` -> ``int`` and
    ``np.floating`` -> ``float`` keeps types meaningful.
    """
    import json as json_mod

    import numpy as np

    from benchmarks.simple_models.run_simple_models_benchmark import save_cache

    results = [
        {
            "dataset": "synth",
            "task": "classification",
            "source": "synthetic",
            # Integer count fields — must remain ints in JSON.
            "n_samples": np.int64(5000),
            "n_features_original": np.int32(7),
            "n_features_tabular": np.int64(20),
            "n_folds": np.int32(5),
            # Large int that would lose precision if cast to float64.
            "huge_int_count": np.int64(2**60 + 1),
            # Float fields — must remain floats.
            "p_value": np.float64(0.03),
            "tabular_improvement_pct": np.float64(2.5),
            # Nested dict with mixed numpy types.
            "metadata": {
                "iter_count": np.int64(42),
                "lr": np.float64(0.01),
                "converged": np.bool_(True),
            },
        }
    ]
    save_cache(results, tmp_path, with_llm=False)
    cache_file = tmp_path / "SIMPLE_MODELS_CACHE.json"
    raw_text = cache_file.read_text(encoding="utf-8")
    # Sanity check the on-disk JSON: integer fields must NOT have a
    # decimal point (e.g. ``"n_samples": 5000`` not ``"n_samples": 5000.0``).
    assert '"n_samples": 5000,' in raw_text or '"n_samples": 5000\n' in raw_text
    assert '"n_folds": 5,' in raw_text or '"n_folds": 5\n' in raw_text
    # Float fields should still serialize with a decimal point.
    assert '"p_value": 0.03' in raw_text

    loaded = json_mod.loads(raw_text)
    row = loaded[0]
    assert row["n_samples"] == 5000 and isinstance(row["n_samples"], int)
    assert row["n_features_original"] == 7 and isinstance(row["n_features_original"], int)
    assert row["n_features_tabular"] == 20 and isinstance(row["n_features_tabular"], int)
    assert row["n_folds"] == 5 and isinstance(row["n_folds"], int)
    # Large int precision preserved (would lose precision through float64).
    assert row["huge_int_count"] == 2**60 + 1 and isinstance(row["huge_int_count"], int)
    # Floats stay floats.
    assert isinstance(row["p_value"], float) and row["p_value"] == 0.03
    assert isinstance(row["tabular_improvement_pct"], float) and row["tabular_improvement_pct"] == 2.5
    # Nested dict preserves int/float/bool distinctions.
    md = row["metadata"]
    assert md["iter_count"] == 42 and isinstance(md["iter_count"], int)
    assert isinstance(md["lr"], float) and md["lr"] == 0.01
    assert md["converged"] is True and isinstance(md["converged"], bool)


def test_preprocess_data_encodes_pandas_string_dtype():
    """Legacy ``preprocess_data`` must encode pandas ``StringDtype`` columns
    (not just ``object``/``category``).

    Regression test for Copilot review on commit 3fdce5e: a previous
    ``include=["object", "category"]`` filter would silently leave
    ``string``/``StringDtype`` columns unencoded, breaking callers that
    rely on the helper's output being model-ready.
    """
    from benchmarks.simple_models.run_simple_models_benchmark import preprocess_data

    X = pd.DataFrame(
        {
            "num": [1.0, 2.0, 3.0, 4.0],
            "obj_cat": ["a", "b", "a", "c"],
            # Pandas StringDtype — would slip through a ``["object", "category"]`` filter.
            "string_ext": pd.array(["x", "y", "x", "z"], dtype="string"),
            # Bool dtype — must remain bool (already numeric-compatible).
            "bool_col": [True, False, True, False],
        }
    )
    y = pd.Series([0, 1, 0, 1])

    X_proc, y_proc = preprocess_data(X, y, task="classification")

    # All non-bool columns must be numeric after preprocessing.
    assert pd.api.types.is_numeric_dtype(X_proc["num"])
    assert pd.api.types.is_integer_dtype(X_proc["obj_cat"])
    assert pd.api.types.is_integer_dtype(
        X_proc["string_ext"]
    ), f"string_ext column was not label-encoded: dtype={X_proc['string_ext'].dtype}"
    # Bool columns pass through unchanged (still boolean / numeric).
    assert pd.api.types.is_bool_dtype(X_proc["bool_col"]) or pd.api.types.is_numeric_dtype(X_proc["bool_col"])


def test_significant_field_is_native_python_bool():
    """``significant = p_value < 0.05`` must yield a native Python ``bool``,
    not ``numpy.bool_``, so the in-memory results dict is immediately
    JSON-/consumer-friendly without relying on ``save_cache`` post-
    processing.

    Regression test for Copilot review on commit 3fdce5e.
    """
    import json as json_mod
    from unittest.mock import patch

    import numpy as np

    from benchmarks.simple_models.run_simple_models_benchmark import run_single_benchmark

    # Use a tiny synthetic dataset so the run is fast; we only care about the
    # type of ``results["significant"]``.
    np.random.seed(0)
    n = 60
    X = pd.DataFrame({"a": np.random.randn(n), "b": np.random.randn(n)})
    y = pd.Series((X["a"] + X["b"] > 0).astype(int), name="target")

    fake_load = lambda name, **kw: (X, y, "classification", "Fake (synthetic)")  # noqa: E731

    with patch("benchmarks.simple_models.run_simple_models_benchmark.load_dataset", fake_load):
        results = run_single_benchmark("titanic", max_features=10, with_llm=False, n_folds=3, n_seeds=1)

    assert results is not None
    assert "significant" in results
    sig = results["significant"]
    # Must be the native ``bool``, not ``numpy.bool_``.
    assert type(sig) is bool, f"significant must be native ``bool``, got {type(sig).__name__}"
    # And must be JSON-serializable directly without any custom encoder.
    json_mod.dumps({"significant": sig})
