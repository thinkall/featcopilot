"""Tests for core module."""

import numpy as np
import pandas as pd
import pytest

from featcopilot.core.base import BaseEngine, BaseSelector
from featcopilot.core.feature import Feature, FeatureOrigin, FeatureSet, FeatureType
from featcopilot.core.registry import FeatureRegistry
from featcopilot.core.transform_rule import TransformRule


class TestFeature:
    """Tests for Feature class."""

    def test_feature_creation(self):
        """Test basic feature creation."""
        feature = Feature(
            name="test_feature",
            dtype=FeatureType.NUMERIC,
            origin=FeatureOrigin.POLYNOMIAL,
            source_columns=["col1"],
            transformation="square",
            explanation="Test explanation",
        )

        assert feature.name == "test_feature"
        assert feature.dtype == FeatureType.NUMERIC
        assert feature.origin == FeatureOrigin.POLYNOMIAL
        assert feature.source_columns == ["col1"]

    def test_feature_to_dict(self):
        """Test feature serialization."""
        feature = Feature(name="test", source_columns=["a", "b"])
        d = feature.to_dict()

        assert d["name"] == "test"
        assert d["source_columns"] == ["a", "b"]

    def test_feature_from_dict(self):
        """Test feature deserialization."""
        d = {
            "name": "test",
            "dtype": "numeric",
            "origin": "polynomial",
            "source_columns": ["x"],
            "transformation": "log",
        }
        feature = Feature.from_dict(d)

        assert feature.name == "test"
        assert feature.dtype == FeatureType.NUMERIC

    def test_feature_compute(self):
        """Test feature computation from code."""
        feature = Feature(name="col_squared", code="result = df['col1'] ** 2")

        df = pd.DataFrame({"col1": [1, 2, 3, 4]})
        result = feature.compute(df)

        assert list(result) == [1, 4, 9, 16]

    def test_feature_compute_uses_safe_builtins(self):
        """Regression: ``Feature.compute`` previously called
        ``exec(code, {"__builtins__": {}}, locals)`` which broke any
        snippet that used a Python builtin (``len``, ``range``, ``int``,
        ``sum``, ...). The fix exposes a curated set of safe builtins
        identical to ``TransformRule._get_safe_builtins`` so common
        idioms work without giving the snippet unrestricted access.
        """
        df = pd.DataFrame({"col1": [10, 20, 30, 40, 50]})

        # ``len`` (most common builtin used in a feature) — would NameError
        # before the fix.
        f_len = Feature(name="row_count", code="result = pd.Series([len(df)] * len(df))")
        result = f_len.compute(df)
        assert list(result) == [5, 5, 5, 5, 5]

        # ``range`` + ``sum`` + numeric builtins — exercises a broader
        # subset of the whitelist in one snippet.
        f_range = Feature(
            name="cumulative_index_sum",
            code="result = pd.Series([sum(range(int(v))) for v in df['col1']])",
        )
        result = f_range.compute(df)
        # sum(range(10))=45, sum(range(20))=190, sum(range(30))=435,
        # sum(range(40))=780, sum(range(50))=1225
        assert list(result) == [45, 190, 435, 780, 1225]

        # ``abs`` + ``round`` + ``min`` / ``max`` — ensure the
        # whitelist's full set is exposed.
        f_clip = Feature(
            name="clipped",
            code="result = pd.Series([round(min(max(abs(v - 25), 0), 20), 2) for v in df['col1']])",
        )
        result = f_clip.compute(df)
        assert list(result) == [15, 5, 5, 15, 20]

    def test_feature_compute_resolves_df_inside_comprehension(self):
        """Regression for round-2 review: when a snippet uses a
        comprehension or lambda whose body references ``df``, ``np`` or
        ``pd``, Python resolves those free variables against the
        enclosing function's *globals*, not the caller's locals. With
        a previous implementation that passed ``df`` only via the
        ``locals`` argument to ``exec`` (and had ``globals`` set to
        only ``__builtins__``), the inner reference would fail with
        ``NameError`` and the call would either crash or be silently
        dropped by ``FeatureSet.compute_all``. The fix uses a single
        shared namespace for both ``globals`` and ``locals``.
        """
        df = pd.DataFrame({"col1": [1, 2, 3, 4, 5]})

        # List comprehension whose body references ``df`` and uses
        # ``range`` / ``len`` builtins — this is the exact pattern
        # called out in the round-2 review.
        f_compre = Feature(
            name="from_comprehension",
            code="result = pd.Series([df['col1'].iloc[i] * 2 for i in range(len(df))])",
        )
        result = f_compre.compute(df)
        assert list(result) == [2, 4, 6, 8, 10]

        # Lambda whose body references ``df`` and ``pd`` — same
        # globals-resolution rule applies.
        f_lambda = Feature(
            name="from_lambda",
            code=(
                "fn = lambda i: int(df['col1'].iloc[i]) ** 2\n" "result = pd.Series([fn(i) for i in range(len(df))])"
            ),
        )
        result = f_lambda.compute(df)
        assert list(result) == [1, 4, 9, 16, 25]

        # Generator expression piped into a builtin — exercises the
        # iterator-protocol path through the same namespace.
        f_gen = Feature(
            name="from_generator",
            code="result = pd.Series([sum(v for v in df['col1'] if v <= n) for n in df['col1']])",
        )
        result = f_gen.compute(df)
        # cumulative sums for thresholds 1, 2, 3, 4, 5: 1, 3, 6, 10, 15
        assert list(result) == [1, 3, 6, 10, 15]

    def test_feature_compute_isolates_safe_builtins_across_calls(self):
        """Regression: ``_SAFE_BUILTINS`` must NOT be passed to ``exec``
        by reference. If it were, a snippet that rebinds entries in its
        own ``__builtins__`` view (e.g. ``__builtins__['len'] = lambda
        x: -999``) would leak that mutation into every subsequent
        ``Feature.compute`` call in the same process, breaking unrelated
        features. The fix passes ``dict(_SAFE_BUILTINS)`` (a fresh
        shallow copy) per call so that side effects are local to that
        single ``exec``.

        Round-2 review note: an earlier version of this test started
        the attacker snippet with ``import builtins``, but ``__import__``
        is intentionally NOT in the safe-builtins whitelist, so the
        snippet raised at the very first line and the surrounding
        ``try/except`` swallowed the failure ─ meaning the test would
        pass even if ``_SAFE_BUILTINS`` were still being shared by
        reference. This rewrite mutates ``__builtins__`` directly via
        the dict that ``exec`` sees as the snippet's builtins source,
        AND asserts inside the snippet (via ``len(df) == -999``) that
        the rebinding actually took effect. That dual assertion is
        what makes the test a real regression check.
        """
        df = pd.DataFrame({"col1": [1, 2, 3]})

        # Attacker snippet: rebind ``len`` in this call's __builtins__
        # dict. The trailing ``result = pd.Series([len(df)])`` would
        # yield 3 if the rebinding had failed, and -999 if the
        # rebinding succeeded. That's the in-snippet proof that
        # mutation IS possible within a single exec ─ which is what
        # makes the per-call-copy isolation contract meaningful.
        f_attacker = Feature(
            name="poisoner",
            code=("__builtins__['len'] = lambda x: -999\nresult = pd.Series([len(df)])\n"),
        )
        poisoned = f_attacker.compute(df)
        assert list(poisoned) == [-999], (
            "Attacker snippet failed to mutate __builtins__: the regression test "
            "is no longer exercising the isolation path it claims to protect"
        )

        # A clean, well-behaved feature run AFTER the attacker must
        # see the original ``len`` builtin, not the poisoned version.
        f_victim = Feature(
            name="clean_user",
            code="result = pd.Series([len(df)] * len(df))",
        )
        clean = f_victim.compute(df)
        assert list(clean) == [3, 3, 3], (
            "Per-call dict copy of _SAFE_BUILTINS is broken: a previous " "feature's mutation leaked into a later call"
        )

    def test_feature_compute_distinguishes_missing_code_from_missing_result(self):
        """Regression: the ``ValueError`` raised by ``Feature.compute``
        must use DIFFERENT messages for the two distinct failure
        modes (``self.code`` empty / missing vs. snippet ran but did
        not bind ``result``). Previously both paths surfaced the
        misleading ``"No code defined ..."`` message even when the
        code was clearly present, making debugging harder.
        """
        # No code at all → "No code defined" branch.
        f_empty = Feature(name="no_code")
        with pytest.raises(ValueError, match="No code defined for feature no_code"):
            f_empty.compute(pd.DataFrame({"x": [1]}))

        # Code present but does not produce ``result`` → distinct
        # message that points at the actual cause.
        f_no_result = Feature(name="no_result", code="x = df['col1'] * 2")
        with pytest.raises(ValueError, match=r"Feature 'no_result' code did not produce a 'result' variable"):
            f_no_result.compute(pd.DataFrame({"col1": [1, 2, 3]}))


class TestFeatureSet:
    """Tests for FeatureSet class."""

    def test_feature_set_add(self):
        """Test adding features to set."""
        fs = FeatureSet()
        f1 = Feature(name="f1")
        f2 = Feature(name="f2")

        fs.add(f1)
        fs.add(f2)

        assert len(fs) == 2
        assert "f1" in fs
        assert "f2" in fs

    def test_feature_set_filter_by_origin(self):
        """Test filtering by origin."""
        fs = FeatureSet(
            [
                Feature(name="f1", origin=FeatureOrigin.POLYNOMIAL),
                Feature(name="f2", origin=FeatureOrigin.INTERACTION),
                Feature(name="f3", origin=FeatureOrigin.POLYNOMIAL),
            ]
        )

        poly_features = fs.filter_by_origin(FeatureOrigin.POLYNOMIAL)
        assert len(poly_features) == 2

    def test_feature_set_get_names(self):
        """Test getting feature names."""
        fs = FeatureSet(
            [
                Feature(name="a"),
                Feature(name="b"),
                Feature(name="c"),
            ]
        )

        names = fs.get_names()
        assert set(names) == {"a", "b", "c"}


class TestFeatureRegistry:
    """Tests for FeatureRegistry."""

    def test_registry_singleton(self):
        """Test registry is singleton."""
        r1 = FeatureRegistry()
        r2 = FeatureRegistry()
        assert r1 is r2

    def test_registry_transformations(self):
        """Test default transformations."""
        registry = FeatureRegistry()
        transforms = registry.list_transformations()

        assert "log" in transforms
        assert "sqrt" in transforms
        assert "square" in transforms

    def test_registry_custom_transformation(self):
        """Test registering custom transformation."""
        registry = FeatureRegistry()
        registry.register_transformation("double", lambda x: x * 2)

        func = registry.get_transformation("double")
        assert func(5) == 10


class TestFeatureComputeNoCode:
    """Tests for Feature.compute when no code is defined."""

    def test_compute_raises_without_code(self):
        """Test that compute raises ValueError when no code is set."""
        feature = Feature(name="no_code_feature")
        df = pd.DataFrame({"col1": [1, 2, 3]})

        with pytest.raises(ValueError, match="No code defined"):
            feature.compute(df)


class TestFeatureSetIteration:
    """Tests for FeatureSet iteration and item access."""

    def _make_feature_set(self):
        """Create a FeatureSet with sample features."""
        return FeatureSet(
            [
                Feature(name="a", importance=0.9, explanation="Explains a"),
                Feature(name="b", importance=0.5, dtype=FeatureType.CATEGORICAL),
                Feature(name="c", importance=0.1),
            ]
        )

    def test_iter(self):
        """Test iterating over FeatureSet yields Feature objects."""
        fs = self._make_feature_set()
        names = [f.name for f in fs]
        assert set(names) == {"a", "b", "c"}

    def test_getitem(self):
        """Test accessing a feature by name with bracket syntax."""
        fs = self._make_feature_set()
        assert fs["a"].name == "a"

    def test_getitem_missing_raises(self):
        """Test KeyError for missing feature name."""
        fs = self._make_feature_set()
        with pytest.raises(KeyError):
            _ = fs["nonexistent"]

    def test_get_existing(self):
        """Test FeatureSet.get returns feature when present."""
        fs = self._make_feature_set()
        feature = fs.get("a")
        assert feature is not None
        assert feature.name == "a"

    def test_get_missing_returns_none(self):
        """Test FeatureSet.get returns None when feature is absent."""
        fs = self._make_feature_set()
        assert fs.get("nonexistent") is None

    def test_remove(self):
        """Test FeatureSet.remove returns the removed feature."""
        fs = self._make_feature_set()
        removed = fs.remove("a")
        assert removed is not None
        assert removed.name == "a"
        assert "a" not in fs

    def test_remove_missing_returns_none(self):
        """Test FeatureSet.remove returns None for missing feature."""
        fs = self._make_feature_set()
        assert fs.remove("nonexistent") is None


class TestFeatureSetFiltering:
    """Tests for FeatureSet filtering and sorting."""

    def test_filter_by_type(self):
        """Test filtering features by FeatureType."""
        fs = FeatureSet(
            [
                Feature(name="num1", dtype=FeatureType.NUMERIC),
                Feature(name="cat1", dtype=FeatureType.CATEGORICAL),
                Feature(name="num2", dtype=FeatureType.NUMERIC),
            ]
        )
        numeric = fs.filter_by_type(FeatureType.NUMERIC)
        assert len(numeric) == 2
        assert "num1" in numeric
        assert "num2" in numeric

    def test_filter_by_importance(self):
        """Test filtering features by minimum importance threshold."""
        fs = FeatureSet(
            [
                Feature(name="high", importance=0.9),
                Feature(name="low", importance=0.1),
                Feature(name="none", importance=None),
            ]
        )
        important = fs.filter_by_importance(0.5)
        assert len(important) == 1
        assert "high" in important

    def test_sort_by_importance_descending(self):
        """Test sorting features by importance in descending order."""
        fs = FeatureSet(
            [
                Feature(name="low", importance=0.1),
                Feature(name="high", importance=0.9),
                Feature(name="mid", importance=0.5),
                Feature(name="none", importance=None),
            ]
        )
        sorted_features = fs.sort_by_importance(descending=True)
        assert len(sorted_features) == 3
        assert sorted_features[0].name == "high"
        assert sorted_features[-1].name == "low"

    def test_sort_by_importance_ascending(self):
        """Test sorting features by importance in ascending order."""
        fs = FeatureSet(
            [
                Feature(name="low", importance=0.1),
                Feature(name="high", importance=0.9),
            ]
        )
        sorted_features = fs.sort_by_importance(descending=False)
        assert sorted_features[0].name == "low"
        assert sorted_features[-1].name == "high"


class TestFeatureSetMergeAndExport:
    """Tests for FeatureSet merge, to_dataframe, and get_explanations."""

    def test_merge(self):
        """Test merging two FeatureSets."""
        fs1 = FeatureSet([Feature(name="a"), Feature(name="b")])
        fs2 = FeatureSet([Feature(name="c"), Feature(name="d")])
        merged = fs1.merge(fs2)
        assert len(merged) == 4
        assert all(n in merged for n in ["a", "b", "c", "d"])

    def test_merge_overwrites_duplicates(self):
        """Test that merge overwrites features with the same name."""
        fs1 = FeatureSet([Feature(name="x", importance=0.1)])
        fs2 = FeatureSet([Feature(name="x", importance=0.9)])
        merged = fs1.merge(fs2)
        assert len(merged) == 1
        assert merged["x"].importance == 0.9

    def test_to_dataframe(self):
        """Test converting FeatureSet to a pandas DataFrame."""
        fs = FeatureSet(
            [
                Feature(name="a", dtype=FeatureType.NUMERIC),
                Feature(name="b", dtype=FeatureType.CATEGORICAL),
            ]
        )
        df = fs.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "name" in df.columns
        assert set(df["name"].tolist()) == {"a", "b"}

    def test_get_explanations(self):
        """Test retrieving explanations from features."""
        fs = FeatureSet(
            [
                Feature(name="a", explanation="Feature A description"),
                Feature(name="b"),
                Feature(name="c", explanation="Feature C description"),
            ]
        )
        explanations = fs.get_explanations()
        assert explanations == {"a": "Feature A description", "c": "Feature C description"}

    def test_get_explanations_empty(self):
        """Test get_explanations when no feature has an explanation."""
        fs = FeatureSet([Feature(name="a"), Feature(name="b")])
        assert fs.get_explanations() == {}


class TestFeatureSetComputeAll:
    """Tests for FeatureSet.compute_all with success and failure cases."""

    def test_compute_all_success(self):
        """Test compute_all adds computed columns to DataFrame."""
        fs = FeatureSet(
            [
                Feature(name="col1_sq", code="result = df['col1'] ** 2"),
                Feature(name="col1_dbl", code="result = df['col1'] * 2"),
            ]
        )
        df = pd.DataFrame({"col1": [1, 2, 3]})
        result = fs.compute_all(df)
        assert list(result["col1_sq"]) == [1, 4, 9]
        assert list(result["col1_dbl"]) == [2, 4, 6]

    def test_compute_all_skips_existing_columns(self):
        """Test compute_all does not overwrite existing columns."""
        fs = FeatureSet([Feature(name="col1", code="result = df['col1'] * 100")])
        df = pd.DataFrame({"col1": [1, 2, 3]})
        result = fs.compute_all(df)
        assert list(result["col1"]) == [1, 2, 3]

    def test_compute_all_skips_no_code(self):
        """Test compute_all skips features without code."""
        fs = FeatureSet([Feature(name="no_code"), Feature(name="sq", code="result = df['col1'] ** 2")])
        df = pd.DataFrame({"col1": [2, 3]})
        result = fs.compute_all(df)
        assert "no_code" not in result.columns
        assert list(result["sq"]) == [4, 9]

    def test_compute_all_handles_failure_gracefully(self):
        """Test compute_all logs warning and continues on failure."""
        fs = FeatureSet(
            [
                Feature(name="bad", code="result = df['nonexistent_col']"),
                Feature(name="good", code="result = df['col1'] + 1"),
            ]
        )
        df = pd.DataFrame({"col1": [10, 20]})
        result = fs.compute_all(df)
        assert "bad" not in result.columns
        assert list(result["good"]) == [11, 21]


class TestRegistryGenerators:
    """Tests for FeatureRegistry generator methods."""

    def test_register_and_get_generator(self):
        """Test registering and retrieving a generator class."""
        registry = FeatureRegistry()

        class DummyGenerator:
            pass

        registry.register_generator("dummy", DummyGenerator)
        assert registry.get_generator("dummy") is DummyGenerator

    def test_get_generator_missing(self):
        """Test get_generator returns None for unknown name."""
        registry = FeatureRegistry()
        assert registry.get_generator("__nonexistent__") is None

    def test_list_generators(self):
        """Test listing registered generator names."""
        registry = FeatureRegistry()
        registry.register_generator("gen_a", type("GenA", (), {}))
        names = registry.list_generators()
        assert "gen_a" in names


class TestRegistryCreateFeature:
    """Tests for FeatureRegistry.create_feature."""

    def test_create_feature_valid(self):
        """Test creating a feature with a valid transformation."""
        registry = FeatureRegistry()
        feature = registry.create_feature(
            name="col1_sqrt",
            transformation="sqrt",
            source_columns=["col1"],
        )
        assert feature.name == "col1_sqrt"
        assert feature.transformation == "sqrt"
        assert feature.origin == FeatureOrigin.POLYNOMIAL

    def test_create_feature_unknown_transformation(self):
        """Test that unknown transformation raises ValueError."""
        registry = FeatureRegistry()
        with pytest.raises(ValueError, match="Unknown transformation"):
            registry.create_feature(
                name="bad",
                transformation="__nonexistent_transform__",
                source_columns=["col1"],
            )


class TestBaseEngineValidation:
    """Tests for BaseEngine input validation and concrete subclass."""

    def test_validate_input_ndarray(self):
        """Test _validate_input converts ndarray to DataFrame."""
        engine = _ConcreteEngine()
        result = engine._validate_input(np.array([[1, 2], [3, 4]]))
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["feature_0", "feature_1"]

    def test_validate_input_invalid_type(self):
        """Test _validate_input raises TypeError for unsupported types."""
        engine = _ConcreteEngine()
        with pytest.raises(TypeError, match="Expected DataFrame or ndarray"):
            engine._validate_input("not a dataframe")

    def test_validate_input_list_raises(self):
        """Test _validate_input raises TypeError for list input."""
        engine = _ConcreteEngine()
        with pytest.raises(TypeError, match="Expected DataFrame or ndarray"):
            engine._validate_input([[1, 2], [3, 4]])

    def test_fit_transform(self):
        """Test fit_transform calls fit then transform."""
        engine = _ConcreteEngine()
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = engine.fit_transform(df)
        assert engine.is_fitted
        assert isinstance(result, pd.DataFrame)

    def test_get_feature_metadata(self):
        """Test get_feature_metadata returns copy of metadata dict."""
        engine = _ConcreteEngine()
        engine._feature_metadata = {"key": "value"}
        meta = engine.get_feature_metadata()
        assert meta == {"key": "value"}
        assert meta is not engine._feature_metadata

    def test_abstract_fit_via_concrete(self):
        """Test abstract fit is callable through concrete subclass."""
        engine = _ConcreteEngine()
        df = pd.DataFrame({"a": [1]})
        returned = engine.fit(df)
        assert returned is engine
        assert engine.is_fitted


class TestBaseSelectorValidation:
    """Tests for BaseSelector input validation and is_fitted property."""

    def test_is_fitted_default(self):
        """Test is_fitted defaults to False."""
        selector = _ConcreteSelector()
        assert selector.is_fitted is False

    def test_is_fitted_after_fit(self):
        """Test is_fitted returns True after fitting."""
        selector = _ConcreteSelector()
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        y = pd.Series([0, 1])
        selector.fit(df, y)
        assert selector.is_fitted is True

    def test_validate_input_ndarray(self):
        """Test _validate_input converts ndarray to DataFrame."""
        selector = _ConcreteSelector()
        result = selector._validate_input(np.array([[1, 2], [3, 4]]))
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["feature_0", "feature_1"]

    def test_validate_input_invalid_type(self):
        """Test _validate_input raises TypeError for unsupported types."""
        selector = _ConcreteSelector()
        with pytest.raises(TypeError, match="Expected DataFrame or ndarray"):
            selector._validate_input("not a dataframe")

    def test_validate_input_dict_raises(self):
        """Test _validate_input raises TypeError for dict input."""
        selector = _ConcreteSelector()
        with pytest.raises(TypeError, match="Expected DataFrame or ndarray"):
            selector._validate_input({"a": [1, 2]})


class TestTransformRuleOutputName:
    """Tests for TransformRule.get_output_name edge cases."""

    def test_output_name_explicit(self):
        """Test get_output_name returns explicit output_name."""
        rule = TransformRule(
            name="test_rule",
            description="Test",
            code="result = df['a']",
            output_name="my_output",
        )
        assert rule.get_output_name() == "my_output"

    def test_output_name_fallback_no_mapping(self):
        """Test get_output_name falls back to rule_{name} without mapping."""
        rule = TransformRule(
            name="test_rule",
            description="Test",
            code="result = df['a']",
        )
        assert rule.get_output_name() == "rule_test_rule"

    def test_output_name_no_input_columns(self):
        """Test get_output_name with mapping but no input_columns."""
        rule = TransformRule(
            name="test_rule",
            description="Test",
            code="result = df['a']",
            input_columns=[],
        )
        assert rule.get_output_name(column_mapping={"a": "b"}) == "rule_test_rule"


class TestTransformRuleMatchesColumns:
    """Tests for TransformRule.matches_columns."""

    def test_matches_empty_input_columns(self):
        """Test matches_columns returns True with empty input_columns."""
        rule = TransformRule(
            name="rule",
            description="Test",
            code="result = 1",
            input_columns=[],
        )
        matches, mapping = rule.matches_columns(["a", "b"])
        assert matches is True
        assert mapping == {}


class TestTransformRuleApply:
    """Tests for TransformRule.apply edge cases."""

    def test_apply_no_result_variable(self):
        """Test apply raises ValueError when code does not set 'result'."""
        rule = TransformRule(
            name="no_result",
            description="Code without result variable",
            code="x = 42",
            input_columns=[],
        )
        df = pd.DataFrame({"a": [1, 2]})
        with pytest.raises(ValueError, match="Rule execution failed"):
            rule.apply(df)

    def test_apply_code_raises_error(self):
        """Test apply raises ValueError on code execution error."""
        rule = TransformRule(
            name="bad_code",
            description="Code that raises",
            code="result = df['nonexistent']",
            input_columns=[],
        )
        df = pd.DataFrame({"a": [1, 2]})
        with pytest.raises(ValueError, match="Rule execution failed"):
            rule.apply(df)

    def test_repr(self):
        """Test TransformRule __repr__ output."""
        rule = TransformRule(
            name="my_rule",
            description="A short description for testing repr output",
            code="result = 1",
        )
        r = repr(rule)
        assert "TransformRule" in r
        assert "my_rule" in r


# ---------------------------------------------------------------------------
# Helper concrete subclasses for abstract base class tests
# ---------------------------------------------------------------------------


class _ConcreteEngine(BaseEngine):
    """Concrete engine for testing BaseEngine abstract methods."""

    def fit(self, X, y=None, **kwargs):
        """Fit the engine."""
        self._is_fitted = True
        return self

    def transform(self, X, **kwargs):
        """Transform data (identity)."""
        return self._validate_input(X)


class _ConcreteSelector(BaseSelector):
    """Concrete selector for testing BaseSelector abstract methods."""

    def fit(self, X, y, **kwargs):
        """Fit the selector."""
        self._is_fitted = True
        return self

    def transform(self, X, **kwargs):
        """Transform data (identity)."""
        return self._validate_input(X)
