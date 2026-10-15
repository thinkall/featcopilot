"""Training-only paired selection and no-feature compute regressions."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge

from featcopilot.engines.predictive import PredictiveTabularEngine


def scores(rows):
    return [PredictiveTabularEngine._summarize_losses(count, losses) for count, losses in rows]


def test_paired_rule_recognizes_consistent_gains_despite_fold_difficulty():
    measured = scores([(0, [1.0, 2.0, 3.0]), (5, [0.9, 1.9, 2.9])])
    assert PredictiveTabularEngine._choose_count(measured) == 0
    assert PredictiveTabularEngine._choose_count(measured, "paired_gain") == 5


def test_paired_rule_rejects_unstable_small_mean_gain():
    measured = scores([(0, [1.0, 1.0, 1.0]), (5, [0.6, 1.2, 1.15])])
    assert PredictiveTabularEngine._choose_count(measured, "paired_gain") == 0


def test_paired_rule_prefers_smaller_equivalent_subsets():
    measured = scores(
        [
            (0, [1.0, 2.0, 3.0]),
            (1, [0.8, 1.8, 2.8]),
            (5, [0.79, 1.81, 2.79]),
        ]
    )
    assert PredictiveTabularEngine._choose_count(measured, "paired_gain") == 1


@pytest.mark.parametrize("rule", ["one_se", "paired_gain"])
def test_no_feature_decision_skips_unused_final_ranking(rule):
    rng = np.random.default_rng(4)
    X = pd.DataFrame(rng.normal(size=(120, 3)), columns=["a", "b", "c"])
    y = rng.normal(size=len(X))
    engine = PredictiveTabularEngine(
        estimator=DummyRegressor(),
        max_candidates=12,
        task="regression",
        stability_repeats=3,
        selection_rule=rule,
    )
    with patch.object(engine, "_rank_candidates", wraps=engine._rank_candidates) as rank:
        engine.fit(X, y)
    assert engine.selection_report_["selected_count"] == 0
    assert rank.call_count == 3
    assert engine.selection_report_["ranking_scope"] == "baseline_retained_no_final_refit"
    assert_frame_equal(engine.transform(X), X)


def test_paired_rule_retains_real_signal_and_batch_invariance():
    rng = np.random.default_rng(18)
    X = pd.DataFrame(rng.normal(size=(300, 2)), columns=["a", "b"])
    y = X.a * X.b + rng.normal(scale=0.05, size=len(X))
    engine = PredictiveTabularEngine(
        estimator=Ridge(),
        task="regression",
        max_candidates=20,
        max_features=8,
        selection_rule="paired_gain",
    )
    result = engine.fit_transform(X, y)
    assert engine.selection_report_["selected_count"] > 0
    assert "paired" in engine.selection_report_["selection_rule"]
    assert_frame_equal(result, pd.concat([engine.transform(X.iloc[:100]), engine.transform(X.iloc[100:])]))


def test_selection_rule_is_validated():
    with pytest.raises(ValueError, match="selection_rule"):
        PredictiveTabularEngine(selection_rule="unverified")
