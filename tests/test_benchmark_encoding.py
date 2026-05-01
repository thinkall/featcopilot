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
