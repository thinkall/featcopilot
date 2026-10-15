"""Pinned, synthetic-only contracts for the separately labeled safe-scale reference."""

import hashlib
import importlib
import inspect
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.utils.validation import check_array

from benchmarks import reference_methods as reference


@pytest.fixture
def compat():
    pytest.importorskip("autofeat")
    module = importlib.import_module("benchmarks._autofeat_safe_scale")
    module.verify_runtime()
    return module


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_constant_one_hot_scaling_is_finite_and_input_is_unchanged(compat, dtype):
    frame = pd.DataFrame({"x": np.arange(12), "category": ["one"] * 12}, index=np.arange(100, 112))
    prepared = reference._PreparedInput(frame)
    train = prepared.transform(frame)
    encoded, _ = reference._autofeat_inputs(prepared, train, train.iloc[:0])
    original = encoded.copy(deep=True)
    values = encoded.to_numpy(dtype=dtype)
    raw = values.copy()
    with np.errstate(invalid="ignore"):
        native = compat.nb_standard_scale(values)
    assert np.isnan(native[:, 1]).all()
    result = compat.scale_features(values)
    np.testing.assert_array_equal(result[:, 0], native[:, 0])
    np.testing.assert_array_equal(result[:, 1], np.zeros(len(frame)))
    np.testing.assert_array_equal(values, raw)
    pd.testing.assert_frame_equal(encoded, original)
    check_array(result)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_nonconstant_scaling_is_exactly_native(compat, dtype):
    values = np.random.default_rng(12).normal(size=(71, 6)).astype(dtype)
    values[:, 2] *= 1e-12
    np.testing.assert_array_equal(compat.scale_features(values), compat.nb_standard_scale(values))


@pytest.mark.parametrize(
    "values",
    [
        [[np.nan, 0.0], [np.nan, 1.0]],
        [[np.nan, 0.0], [1.0, 1.0]],
        [[np.inf, 0.0], [np.inf, 1.0]],
        [[-np.inf, 0.0], [1.0, 1.0]],
    ],
)
def test_missing_and_infinite_values_are_not_repaired(compat, values):
    array = np.array(values)
    with np.errstate(invalid="ignore", divide="ignore"):
        native = compat.nb_standard_scale(array)
        result = compat.scale_features(array)
    np.testing.assert_array_equal(result, native)
    with pytest.raises(ValueError):
        check_array(result)


def test_nonconstant_std_underflow_is_not_repaired(compat):
    values = np.array([[0.0], [1e-200]], dtype=np.float64)
    assert values[0, 0] != values[1, 0]
    assert compat.nb_nanstd(values, 0)[0] == 0
    with np.errstate(invalid="ignore", divide="ignore"):
        native = compat.nb_standard_scale(values)
        result = compat.scale_features(values)
    np.testing.assert_array_equal(result, native)
    assert not np.isfinite(result).all()


def test_shuffled_noise_uses_private_scaler_preserves_rng_and_shape(compat):
    from autofeat import featsel

    x = np.ones((60, 2))
    with reference._seeded(19), np.errstate(invalid="ignore"):
        native = featsel._add_noise_features(x)
        native_rng = pickle.dumps(np.random.get_state())
    with reference._seeded(19):
        fixed = compat._add_noise_features(x)
        fixed_rng = pickle.dumps(np.random.get_state())
    assert fixed.shape == native.shape == (60, 7)
    np.testing.assert_array_equal(fixed[:, :2], x)
    np.testing.assert_array_equal(fixed[:, 2:4], np.zeros((60, 2)))
    np.testing.assert_array_equal(fixed[:, 4:], native[:, 4:])
    assert np.isnan(native[:, 2:4]).all()
    assert fixed_rng == native_rng


@pytest.mark.parametrize("constant_target", [False, True])
def test_target_scaling_stays_native_and_unmasked(compat, monkeypatch, constant_target):
    captured = {}

    def stop(frame, target, problem_type, verbose):
        captured["features"] = frame.to_numpy()
        captured["target"] = target
        raise RuntimeError("stop before estimator fitting")

    monkeypatch.setattr(compat, "_select_features_1run", stop)
    x = pd.DataFrame({"constant": np.ones(20), "varying": np.arange(20)})
    target = np.ones(20) if constant_target else np.arange(20, dtype=float)
    with reference._seeded(12), pytest.raises(RuntimeError, match="stop before"):
        compat._select_features(x, target, problem_type="regression", n_jobs=1)
    assert np.isfinite(captured["features"]).all()
    rows = np.random.RandomState(0).permutation(20)[:17]
    with np.errstate(invalid="ignore"):
        expected = compat.nb_standard_scale(target.reshape(-1, 1)).ravel()[rows]
    np.testing.assert_array_equal(captured["target"], expected)
    assert (
        compat._select_features.__globals__["nb_standard_scale"]
        is importlib.import_module("autofeat.nb_utils").nb_standard_scale
    )


def test_runtime_version_and_source_pins_fail_closed(compat, monkeypatch):
    versions = compat.version
    monkeypatch.setattr(compat, "version", lambda name: "2.1.3" if name == "autofeat" else versions(name))
    with pytest.raises(compat.AutoFeatCompatibilityError, match="autofeat"):
        compat.verify_runtime()
    monkeypatch.setattr(compat, "version", versions)
    monkeypatch.setattr(compat, "_source_digest", lambda path: "changed")
    with pytest.raises(compat.AutoFeatCompatibilityError, match="source"):
        compat.verify_runtime()


def test_source_and_bindings_are_private(compat):
    from autofeat import autofeat, featsel, nb_utils

    assert compat._fit_transform.__globals__["_select_features"] is compat._select_features
    assert compat._select_features.__globals__["_select_features_1run"] is compat._select_features_1run
    assert compat._noise_filtering.__globals__["_add_noise_features"] is compat._add_noise_features
    assert compat._add_noise_features.__globals__["scale_features"] is compat.scale_features
    assert autofeat.AutoFeatModel.fit_transform.__globals__["select_features"] is featsel.select_features
    assert featsel.select_features.__globals__["nb_standard_scale"] is nb_utils.nb_standard_scale
    assert featsel._add_noise_features.__globals__["nb_standard_scale"] is nb_utils.nb_standard_scale
    assert compat.engineer_features is autofeat.engineer_features
    for filename, expected in compat.SOURCE_SHA256.items():
        source = Path(inspect.getfile(autofeat)).with_name(filename)
        assert hashlib.sha256(source.read_bytes()).hexdigest() == expected


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_constructor_and_unfitted_pickle_preserve_upstream_defaults(compat, task):
    from autofeat import AutoFeatClassifier, AutoFeatRegressor

    cls = compat.SafeScaleAutoFeatClassifier if task == "classification" else compat.SafeScaleAutoFeatRegressor
    native = AutoFeatClassifier if task == "classification" else AutoFeatRegressor
    model = cls(max_gb=None, n_jobs=1, verbose=0)
    assert model.get_params() == native(max_gb=None, n_jobs=1, verbose=0).get_params()
    assert pickle.loads(pickle.dumps(model)).get_params() == model.get_params()


def test_reference_integration_is_explicit_and_native_routes_unchanged(monkeypatch, tmp_path):
    calls = []

    class Engine:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def fit_transform(self, frame, labels):
            return frame.assign(**{f"selected_{i}": frame.iloc[:, 0] ** 2 + i for i in range(55)})

        def transform(self, frame):
            return frame.assign(**{f"selected_{i}": frame.iloc[:, 0] ** 2 + i for i in range(55)})

    fake = type(
        "Compatibility",
        (),
        {
            "SafeScaleAutoFeatClassifier": Engine,
            "SafeScaleAutoFeatRegressor": Engine,
            "provenance": staticmethod(lambda: {"name": "safe-scale-v1", "target_scaling": "unchanged"}),
        },
    )
    original = reference.import_module
    imports = []

    def load(name):
        imports.append(name)
        return fake if name == "benchmarks._autofeat_safe_scale" else original(name)

    monkeypatch.setattr(reference, "import_module", load)
    monkeypatch.setattr(reference, "_load_autofeat", lambda task: Engine)
    monkeypatch.setattr(reference, "version", lambda name: "test")
    frame = pd.DataFrame({"x": np.arange(12, dtype=float)}, index=np.arange(100, 112))
    labels = pd.Series(np.arange(12, dtype=float), index=frame.index)
    for method, cap in (("autofeat-native", None), ("autofeat-default", 1), ("autofeat-native-safe-scale-v1", None)):
        imports.clear()
        train, test, info = reference.generate_reference(
            frame,
            frame,
            labels,
            method=method,
            task="regression",
            max_added_features=None,
            threads=1,
            seed=7,
            workspace=tmp_path,
        )
        assert calls[-1] == {"max_gb": cap, "n_jobs": 1, "verbose": 0}
        assert info["method"] == method
        assert info["width_policy"] == "native_output"
        assert info["added_features"] == 55
        assert info["resources"]["max_gb"] == cap
        assert info["resources"]["hard_memory_limit"] is False
        pd.testing.assert_frame_equal(train[frame.columns], frame)
        assert train.columns.equals(test.columns)
        if method.endswith("safe-scale-v1"):
            assert imports == ["benchmarks._autofeat_safe_scale"]
            assert info["search"]["compatibility"]["target_scaling"] == "unchanged"
        else:
            assert imports == []
            assert "compatibility" not in info["search"]


@pytest.mark.parametrize("cap", [0, 1, 50])
def test_compatibility_rejects_width_caps_before_import(tmp_path, cap, monkeypatch):
    monkeypatch.setattr(reference, "import_module", lambda name: pytest.fail("must reject before import"))
    frame = pd.DataFrame({"x": np.arange(12)})
    with pytest.raises(reference.ReferenceWidthError, match="safe-scale-v1"):
        reference.generate_reference(
            frame,
            frame,
            frame.x,
            method="autofeat-native-safe-scale-v1",
            task="regression",
            max_added_features=cap,
            threads=1,
            seed=7,
            workspace=tmp_path,
        )


def test_reference_pin_failure_is_not_a_native_fallback(tmp_path, monkeypatch):
    def fail_pin():
        raise ImportError("source mismatch")

    monkeypatch.setattr(reference, "import_module", lambda name: type("Module", (), {"provenance": fail_pin}))
    monkeypatch.setattr(reference, "_load_autofeat", lambda task: pytest.fail("no native fallback"))
    frame = pd.DataFrame({"x": np.arange(12)})
    with pytest.raises(reference.ReferenceDependencyError, match="source mismatch"):
        reference.generate_reference(
            frame,
            frame,
            frame.x,
            method="autofeat-native-safe-scale-v1",
            task="regression",
            max_added_features=None,
            threads=1,
            seed=7,
            workspace=tmp_path,
        )


def test_spawn_importability_and_bounded_synthetic_fit(compat):
    script = r"""
import importlib, json, multiprocessing as mp, pickle, warnings
import numpy as np
import pandas as pd
import psutil
from concurrent.futures import ProcessPoolExecutor
from threadpoolctl import threadpool_limits
from benchmarks.reference_methods import _seeded

native = importlib.import_module("autofeat.autofeat")
selector = importlib.import_module("autofeat.featsel")
identities = (native.AutoFeatModel.fit_transform, native.select_features, selector.nb_standard_scale,
              selector._add_noise_features, np.nanstd, np.nanmean)
from benchmarks import _autofeat_safe_scale as c
assert (native.AutoFeatModel.fit_transform, native.select_features, selector.nb_standard_scale,
        selector._add_noise_features, np.nanstd, np.nanmean) == identities
process = psutil.Process()
process.cpu_affinity([process.cpu_affinity()[0]])
with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn")) as pool:
    future = pool.submit(c.scale_features, np.column_stack([np.arange(12.), np.ones(12)]))
    assert np.isfinite(future.result(timeout=45)).all()
    assert pool.submit(c.provenance).result(timeout=45)["name"] == "safe-scale-v1"
    spawned = pool.submit(c.SafeScaleAutoFeatClassifier, max_gb=None, n_jobs=1).result(timeout=45)
    assert type(spawned) is c.SafeScaleAutoFeatClassifier

x = pd.DataFrame({"x000000": np.linspace(0.25, 2.0, 24), "cat_x000001_0": np.ones(24)})
y = np.arange(24) % 2
original = x.copy()
with threadpool_limits(limits=1), _seeded(19), warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    model = c.SafeScaleAutoFeatClassifier(max_gb=None, n_jobs=1, verbose=0)
    out = model.fit_transform(x, y)
    transformed = model.transform(x.iloc[:3])
assert len(out) == len(x) and transformed.columns.equals(out.columns)
pd.testing.assert_frame_equal(x, original)
assert (native.AutoFeatModel.fit_transform, native.select_features, selector.nb_standard_scale,
        selector._add_noise_features, np.nanstd, np.nanmean) == identities
restored = pickle.loads(pickle.dumps(model))
pd.testing.assert_frame_equal(restored.transform(x.iloc[:3]), transformed)
print(json.dumps({"rows": len(out), "columns": len(out.columns), "spawn": "passed", "pickle": "passed"}))
"""
    environment = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMBA_NUM_THREADS": "1",
    }
    result = subprocess.run(
        [sys.executable, "-c", script], env=environment, capture_output=True, text=True, timeout=110, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1])["spawn"] == "passed"
