"""Tests for core module."""

import pandas as pd

from featcopilot.core.feature import Feature, FeatureOrigin, FeatureSet, FeatureType
from featcopilot.core.registry import FeatureRegistry


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
