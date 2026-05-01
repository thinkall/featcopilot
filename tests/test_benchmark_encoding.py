"""Tests for the benchmark non-numeric encoding helper."""

from __future__ import annotations

import numpy as np
import pandas as pd

from benchmarks.simple_models.run_simple_models_benchmark import _label_encode_non_numeric


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
