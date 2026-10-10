"""Inductive, paired tabular feature-engineering benchmark.

Run ``python -m benchmarks.tabular --help``. See ``benchmarks/README.md``
for the protocol, comparator restrictions, and interpretation of results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import pickle
import random
import subprocess
import sys
import time
import urllib.request
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, PolynomialFeatures, StandardScaler

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent
SUITE_PATH = Path(__file__).with_name("tabular_suite.json")
PROTOCOL_VERSION = 3
METHODS = (
    "baseline",
    "baseline-robust",
    "baseline-target-robust",
    "baseline-discovery-robust",
    "featcopilot",
    "featcopilot-task",
    "featcopilot-predictive",
    "featcopilot-paired",
    "featcopilot-extended",
    "featcopilot-extended-paired",
    "featcopilot-importance",
    "featcopilot-importance-paired",
    "featcopilot-frequency-paired",
    "featcopilot-frequency-aligned",
    "featcopilot-frequency-aligned-robust",
    "featcopilot-frequency-aligned-target-robust",
    "featcopilot-regression-average-robust",
    "featcopilot-bagged-aligned-robust",
    "featcopilot-plan-gated-robust",
    "featcopilot-unselected",
    "featcopilot-selection-only",
    "polynomial",
    "openfe-stateless",
    "autofeat",
    "openfe-inductive",
    "openfe-inductive-native",
    "openfe-inductive-native-robust",
    "autofeat-default",
    "autofeat-native",
    "autofeat-native-robust",
    "autofeat-native-safe-scale-v1-robust",
    "llm-plan",
    "llm-hybrid",
    "llm-hybrid-namespaced",
)
DEFAULT_METHODS = (
    "baseline",
    "featcopilot",
    "featcopilot-task",
    "featcopilot-predictive",
    "polynomial",
    "openfe-stateless",
    "autofeat",
)
PREDICTIVE_METHODS = {
    "featcopilot-predictive",
    "featcopilot-paired",
    "featcopilot-extended",
    "featcopilot-extended-paired",
    "featcopilot-importance",
    "featcopilot-importance-paired",
    "featcopilot-frequency-paired",
    "featcopilot-frequency-aligned",
    "featcopilot-frequency-aligned-robust",
    "featcopilot-frequency-aligned-target-robust",
    "featcopilot-bagged-aligned-robust",
}
ALIGNED_METHODS = {
    "featcopilot-frequency-aligned",
    "featcopilot-frequency-aligned-robust",
    "featcopilot-frequency-aligned-target-robust",
    "featcopilot-bagged-aligned-robust",
}
PLAN_VALIDATION_METHODS = {"baseline-discovery-robust", "featcopilot-plan-gated-robust"}
ROBUST_METHODS = {
    "baseline-robust",
    "baseline-target-robust",
    "featcopilot-frequency-aligned-robust",
    "featcopilot-frequency-aligned-target-robust",
    "featcopilot-regression-average-robust",
    "featcopilot-bagged-aligned-robust",
    "openfe-inductive-native-robust",
    "autofeat-native-robust",
    "autofeat-native-safe-scale-v1-robust",
} | PLAN_VALIDATION_METHODS
ROBUST_REFERENCE_METHODS = {
    "openfe-inductive-native-robust",
    "autofeat-native-robust",
    "autofeat-native-safe-scale-v1-robust",
}
TARGET_ENCODING_METHODS = {"baseline-target-robust", "featcopilot-frequency-aligned-target-robust"}
ROBUST_CLIP_QUANTILES = (0.0, 1.0)
LEARNERS = ("linear", "random_forest", "lightgbm", "catboost", "xgboost")
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "LOKY_MAX_CPU_COUNT",
)


def digest(data: bytes) -> str:
    """Return a reproducibility fingerprint."""
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, data: Any) -> None:
    """Atomically persist standards-compliant JSON."""
    from benchmarks.run_state import atomic_json

    atomic_json(path, data)


def load_real_dataset(spec: dict, revision: str, cache: Path, max_rows: int) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Load an immutable public CSV snapshot; never substitute synthetic data."""
    relative = Path(spec["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Dataset path must be relative to the pinned snapshot")
    path = cache / revision / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/datasets/inria-soda/tabular-benchmark/resolve/{revision}/{spec['path']}"
    if not path.exists():
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
        temporary = path.with_suffix(".tmp")
        temporary.write_bytes(data)
        temporary.replace(path)
    data = path.read_bytes()
    frame = pd.read_csv(path)
    if spec["target"] not in frame:
        raise ValueError(f"Missing declared target {spec['target']!r}")
    if frame[spec["target"]].isna().any():
        raise ValueError("Missing targets require an explicit dataset policy")
    full_rows = len(frame)
    row_indices = np.arange(full_rows)
    if max_rows and full_rows > max_rows:
        # Selection is fixed before outer CV and is shared by every method.
        row_indices = np.sort(np.random.default_rng(2026).choice(full_rows, max_rows, replace=False))
        frame = frame.iloc[row_indices].copy()
    y = frame.pop(spec["target"]).reset_index(drop=True)
    for column in spec["categorical"]:
        frame[column] = frame[column].astype("category")
    frame = frame.reset_index(drop=True)
    original_columns = frame.columns.tolist()
    # All methods receive the identical collision-free names.
    frame.columns = [f"x{i}" for i in range(frame.shape[1])]
    if spec["task"] == "classification":
        encoder = LabelEncoder()
        y = pd.Series(encoder.fit_transform(y))
    elif not np.isfinite(np.asarray(y, dtype=float)).all():
        raise ValueError("Regression targets must be finite")
    metadata = {
        **spec,
        "source": "real_world",
        "url": url,
        "sha256": digest(data),
        "original_rows": full_rows,
        "sampled_rows": len(frame),
        "sample_indices_sha256": digest(row_indices.astype("<i8").tobytes()),
        "columns": original_columns,
        "dtypes": [str(dtype) for dtype in frame.dtypes],
    }
    return frame, y, metadata


def prepare_numeric(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    impute: bool = True,
    preserve_boolean: bool = False,
    preserve_categorical_types: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize schemas and optionally impute numeric inputs.

    Parameters
    ----------
    train, test : pandas.DataFrame
        Training and held-out frames with matching columns.
    impute : bool, default=True
        Fill numeric missing values using the supplied training frame. Disable
        before methods that create their own internal training partitions.
    preserve_boolean : bool, default=False
        Keep Boolean columns outside numeric clipping in robust variants.
    preserve_categorical_types : bool, default=False
        Leave categorical scalars and dtypes intact for the typed target encoder.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        Independent frames with finite-or-missing numeric inputs and explicit
        categorical missing-value markers.
    """
    if not train.columns.equals(test.columns):
        raise ValueError("Train/test feature schemas differ")
    train, test = train.copy(), test.copy()
    for col in train:
        if preserve_boolean and pd.api.types.is_bool_dtype(train[col]):
            continue
        if pd.api.types.is_numeric_dtype(train[col]):
            train[col] = train[col].astype(float).replace([np.inf, -np.inf], np.nan)
            test[col] = test[col].astype(float).replace([np.inf, -np.inf], np.nan)
            if impute:
                median = train[col].median() if train[col].notna().any() else 0.0
                train[col] = train[col].fillna(median)
                test[col] = test[col].fillna(median)
        else:
            if preserve_categorical_types:
                continue
            # Preserve missingness separately from literal strings.
            train[col] = train[col].astype(object).map(lambda value: "missing:" if pd.isna(value) else f"value:{value}")
            test[col] = test[col].astype(object).map(lambda value: "missing:" if pd.isna(value) else f"value:{value}")
    return train, test


def generate_features(
    train: pd.DataFrame, test: pd.DataFrame, y: pd.Series, method: str, task: str, config: dict, workspace: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit a method on training rows only and preserve original columns."""
    if method in PLAN_VALIDATION_METHODS:
        raise ValueError(f"{method} requires run_case for its complete discovery/gate/refit protocol")
    if method in TARGET_ENCODING_METHODS:
        if task != "classification":
            raise ValueError("Target encoding benchmark methods support classification only")
        if any(config.get(name) is not None for name in ("groups", "timestamps", "label_end_times")):
            raise ValueError("Target encoding benchmark methods support IID only")
        config = {**config, "categorical_encoding": "target"}
    if method in ROBUST_METHODS:
        config = {**config, "numeric_clip_quantiles": ROBUST_CLIP_QUANTILES}
    nested_preprocessing = method in PREDICTIVE_METHODS | ROBUST_METHODS | {
        "llm-plan",
        "llm-hybrid",
        "llm-hybrid-namespaced",
    }
    train, test = prepare_numeric(
        train,
        test,
        impute=not nested_preprocessing,
        preserve_boolean=method in ROBUST_METHODS,
        preserve_categorical_types=method in TARGET_ENCODING_METHODS,
    )
    if method in {"baseline", "baseline-robust", "baseline-target-robust"}:
        return train, test
    max_added = config["max_added_features"]
    numeric = train.select_dtypes(include=np.number).columns.tolist()
    if method in {"llm-plan", "llm-hybrid", "llm-hybrid-namespaced"}:
        from benchmarks.llm_evaluation import CopilotPlanProvider, EvaluationBudget, EvaluationLedger, evaluate_plan_fit
        from featcopilot import AutoFeatureEngineer
        from featcopilot.utils.models import DEFAULT_MODEL
        from featcopilot.utils.task import resolve_task

        if not config.get("live_llm") or not config.get("llm_ledger"):
            raise ValueError("Live LLM methods require explicit approval and a shared call-budget ledger")
        budget = EvaluationBudget(
            max_calls=100, max_tokens=4096, require_hard_token_cap=not config.get("allow_soft_llm_tokens", False)
        )
        ledger = EvaluationLedger(config["llm_ledger"], budget)
        provider = CopilotPlanProvider(model=DEFAULT_MODEL, timeout=60.0)
        deterministic_factory = None
        if method != "llm-plan":
            learner = config["selection_learner"]
            template = make_model(train, task, learner, candidate_parameters(learner)[1], config).steps[-1][1]
            if learner == "catboost":
                template = type(template)(
                    **{key: value for key, value in template.get_params().items() if key != "cat_features"}
                )

            def deterministic_factory():
                return AutoFeatureEngineer(
                    selection_strategy="predictive",
                    task=resolve_task(y, task),
                    selection_estimator=template,
                    max_features=train.shape[1] + max_added,
                    max_candidates=config.get("max_candidates", 1000),
                    stability_repeats=config.get("stability_repeats", 3),
                    screening=config.get("screening", "auto"),
                    random_state=config["seed"],
                    gate_n_jobs=config["threads"],
                    leakage_guard="off",
                )

        result = evaluate_plan_fit(
            train,
            y,
            test,
            mode="typed_llm" if method == "llm-plan" else "hybrid",
            provider=provider,
            ledger=ledger,
            live_approved=True,
            deterministic_factory=deterministic_factory,
            composition_policy="namespace" if method == "llm-hybrid-namespaced" else "raise",
            task=resolve_task(y, task),
            random_state=config["seed"],
            column_descriptions=config.get("column_descriptions"),
            max_rules=8,
            task_description=f"Generate generalizable {task} features using only the declared training schema.",
            fit_id=f"{workspace.name}:{len(train)}",
        )
        result.train_X.attrs["llm_diagnostics"] = asdict(result.diagnostics)
        if result.plan_engine is not None:
            stored = result.plan_engine.to_dict()["rules"]
            if not isinstance(stored, list):
                raise ValueError("Fitted plan export must contain a rule list")
            definitions = []
            for entry in stored:
                if not isinstance(entry, dict) or not isinstance(entry.get("rule"), dict):
                    raise ValueError("Fitted plan export contains an invalid rule definition")
                definitions.append(entry["rule"])
            result.train_X.attrs["llm_plan"] = definitions
        return result.train_X, result.test_X
    if (
        method
        in {"openfe-inductive", "openfe-inductive-native", "autofeat-default", "autofeat-native"}
        | ROBUST_REFERENCE_METHODS
    ):
        from benchmarks.reference_methods import generate_reference

        if method in ROBUST_REFERENCE_METHODS:
            reference_train, reference_test = prepare_numeric(train, test, impute=True)
            # Categories are already normalized above; only restore numeric
            # original columns after the reference's unchanged generation path.
            categorical = [column for column in train if not pd.api.types.is_numeric_dtype(train[column])]
            reference_train[categorical] = train[categorical]
            reference_test[categorical] = test[categorical]
        else:
            reference_train, reference_test = train, test
        output_train, output_test, diagnostics = generate_reference(
            reference_train,
            reference_test,
            y,
            method="openfe-inductive" if method.startswith("openfe-inductive") else method.removesuffix("-robust"),
            task=task,
            max_added_features=max_added if method == "openfe-inductive" else None,
            threads=config["threads"],
            seed=config["seed"],
            workspace=workspace,
        )
        if method in ROBUST_REFERENCE_METHODS:
            output_train = output_train.copy()
            output_test = output_test.copy()
            for column in train:
                output_train[column] = train[column]
                output_test[column] = test[column]
            diagnostics = {
                **diagnostics,
                "numeric_clip_quantiles": list(ROBUST_CLIP_QUANTILES),
                "clipping_scope": "final_matched_learner_only",
                "generation_preprocessing": "existing_training_median_fill",
                "original_numeric_missingness_restored": True,
            }
        if not config.get("detailed_diagnostics", False):
            preprocessing = dict(diagnostics.get("preprocessing", {}))
            preprocessing.pop("one_hot_levels", None)
            diagnostics = {**diagnostics, "preprocessing": preprocessing}
        output_train.attrs["reference_diagnostics"] = diagnostics
        return output_train, output_test
    if (
        method
        in {"featcopilot", "featcopilot-task", "featcopilot-unselected", "featcopilot-selection-only"}
        | PREDICTIVE_METHODS
    ):
        from featcopilot import AutoFeatureEngineer

        predictive = method in PREDICTIVE_METHODS
        selection_estimator = None
        if predictive:
            learner = config["selection_learner"]
            selection_parameters = candidate_parameters(learner)[1]
            if method in ALIGNED_METHODS:
                selection_parameters = config.get("selection_model_parameters", selection_parameters)
                if selection_parameters not in candidate_parameters(learner):
                    raise ValueError("Aligned selection parameters must belong to the declared learner search")
            selection_estimator = make_model(train, task, learner, selection_parameters, config).steps[-1][1]
            if learner == "catboost":
                parameters = {
                    key: value for key, value in selection_estimator.get_params().items() if key != "cat_features"
                }
                selection_estimator = type(selection_estimator)(**parameters)
        engineer = AutoFeatureEngineer(
            engines=["tabular"],
            max_features=train.shape[1] + max_added,
            gate_n_jobs=config["threads"],
            leakage_guard="off",
            task=task if method != "featcopilot" else "auto",
            selection_strategy="predictive" if predictive else "legacy",
            selection_estimator=selection_estimator,
            max_candidates=config.get("max_candidates", 1000),
            random_state=config["seed"],
            stability_repeats=config.get("stability_repeats", 3),
            screening=(
                "importance"
                if method
                in {
                    "featcopilot-importance",
                    "featcopilot-importance-paired",
                    "featcopilot-frequency-paired",
                    "featcopilot-frequency-aligned",
                    "featcopilot-frequency-aligned-robust",
                    "featcopilot-frequency-aligned-target-robust",
                    "featcopilot-bagged-aligned-robust",
                }
                else config.get("screening", "auto")
            ),
            selection_rule=(
                "paired_gain"
                if method
                in {
                    "featcopilot-paired",
                    "featcopilot-extended-paired",
                    "featcopilot-importance-paired",
                    "featcopilot-frequency-paired",
                    "featcopilot-frequency-aligned",
                    "featcopilot-frequency-aligned-robust",
                    "featcopilot-frequency-aligned-target-robust",
                    "featcopilot-bagged-aligned-robust",
                }
                else "one_se"
            ),
            candidate_set=(
                "extended"
                if method
                in {
                    "featcopilot-extended",
                    "featcopilot-extended-paired",
                    "featcopilot-importance",
                    "featcopilot-importance-paired",
                    "featcopilot-frequency-paired",
                    "featcopilot-frequency-aligned",
                    "featcopilot-frequency-aligned-robust",
                    "featcopilot-frequency-aligned-target-robust",
                    "featcopilot-bagged-aligned-robust",
                }
                else "standard"
            ),
            include_frequency=method in {"featcopilot-frequency-paired"} | ALIGNED_METHODS,
            numeric_clip_quantiles=config.get("numeric_clip_quantiles"),
            importance_resamples=5 if method == "featcopilot-bagged-aligned-robust" else 1,
            categorical_encoding=config.get("categorical_encoding", "one_hot"),
        )
        fit_metadata = {}
        for name in ("groups", "timestamps", "label_end_times"):
            values = config.get(name)
            if values is not None:
                if not predictive:
                    raise ValueError(f"{method} does not support {name}-aware internal selection")
                fit_metadata[name] = values.loc[train.index].to_numpy()
        transformed_train = engineer.fit_transform(
            train,
            y,
            apply_selection=method != "featcopilot-unselected",
            apply_gate=method != "featcopilot-selection-only",
            **fit_metadata,
        )
        transformed_test = engineer.transform(test)
        if engineer.selection_report_ is not None:
            transformed_train.attrs["selection_report"] = engineer.selection_report_
        return transformed_train, transformed_test
    if not numeric:
        raise ValueError(f"{method} requires numeric features")
    if method == "polynomial":
        polynomial = PolynomialFeatures(degree=2, include_bias=False)
        generated_train = polynomial.fit_transform(train[numeric])[:, len(numeric) :]
        generated_test = polynomial.transform(test[numeric])[:, len(numeric) :]
        selector = SelectKBest(
            f_classif if task == "classification" else f_regression, k=min(max_added, generated_train.shape[1])
        )
        generated_train = selector.fit_transform(generated_train, y)
        generated_test = selector.transform(generated_test)
    elif method == "autofeat":
        from autofeat import AutoFeatClassifier, AutoFeatRegressor

        model_class = AutoFeatClassifier if task == "classification" else AutoFeatRegressor
        engineer = model_class(
            feateng_steps=2,
            featsel_runs=3,
            max_gb=1,
            transformations=("1/", "log", "sqrt", "^2"),
            n_jobs=config["threads"],
            verbose=0,
        )
        engineered_train = engineer.fit_transform(train[numeric], np.asarray(y))
        engineered_test = engineer.transform(test[numeric])
        generated_train = np.asarray(engineered_train)[:, len(numeric) : len(numeric) + max_added]
        generated_test = np.asarray(engineered_test)[:, len(numeric) : len(numeric) + max_added]
    elif method == "openfe-stateless":
        from openfe import get_candidate_features

        candidates = get_candidate_features(numerical_features=numeric, order=1)
        # OpenFE's public transform concatenates test rows for global operators.
        # This explicitly named variant excludes those operators, not test leakage.
        allowed = {"abs", "log", "sqrt", "square", "sigmoid", "round", "residual", "max", "min", "+", "-", "*", "/"}
        candidates = [feature for feature in candidates if feature.name in allowed]
        engineer = make_openfe()
        selected = engineer.fit(
            train[numeric].copy(),
            pd.DataFrame(np.asarray(y), index=train.index, columns=["target"]),
            task=task,
            candidate_features_list=candidates,
            n_jobs=config["threads"],
            seed=config["seed"],
            tmp_save_path=str(workspace / "openfe.feather"),
            verbose=False,
        )[:max_added]
        generated_train = (
            np.column_stack([feature.calculate(train[numeric]) for feature in selected])
            if selected
            else np.empty((len(train), 0))
        )
        generated_test = (
            np.column_stack([feature.calculate(test[numeric]) for feature in selected])
            if selected
            else np.empty((len(test), 0))
        )
    else:
        raise ValueError(f"Unknown method {method!r}")
    names = [f"derived_{i}" for i in range(generated_train.shape[1])]
    return (
        pd.concat([train, pd.DataFrame(generated_train, index=train.index, columns=names)], axis=1),
        pd.concat([test, pd.DataFrame(generated_test, index=test.index, columns=names)], axis=1),
    )


def make_openfe():
    """Adapt OpenFE 0.0.12's removed sklearn RMSE keyword without changing its metric."""
    from benchmarks.tabular_openfe import CompatibleOpenFE

    return CompatibleOpenFE()


def candidate_parameters(learner: str) -> list[dict]:
    """Return the predeclared equal-per-method learner search spaces."""
    if learner == "linear":
        return [{"regularization": value} for value in (0.1, 1.0, 10.0)]
    if learner == "random_forest":
        return [{"max_depth": depth} for depth in (8, 16, None)]
    if learner == "lightgbm":
        return [{"num_leaves": leaves} for leaves in (7, 15, 31)]
    if learner in {"catboost", "xgboost"}:
        return [{"depth": depth} for depth in (3, 6, 9)]
    raise ValueError(f"Unknown learner {learner!r}")


class NativeCategoricalFrame(BaseEstimator, TransformerMixin):
    """Retain dataframe/category identity for matched native CatBoost pipelines."""

    def __init__(self, boolean_as_numeric: bool = False):
        self.boolean_as_numeric = boolean_as_numeric

    def fit(self, X, y=None):
        self.columns_ = list(X.columns)
        self.categorical_ = list(X.select_dtypes(exclude=np.number).columns)
        if self.boolean_as_numeric:
            self.categorical_ = [column for column in self.categorical_ if not pd.api.types.is_bool_dtype(X[column])]
        return self

    def transform(self, X):
        if list(X.columns) != self.columns_:
            raise ValueError("Native model preprocessing requires the fitted feature schema")
        result = X.copy()
        for column in self.columns_:
            if column in self.categorical_:
                result[column] = (
                    result[column].astype(object).map(lambda value: "missing:" if pd.isna(value) else f"value:{value}")
                )
            else:
                result[column] = result[column].astype(float).replace([np.inf, -np.inf], np.nan)
        return result


def selection_diagnostics(report: dict | None, detailed: bool) -> dict | None:
    """Keep standard result rows compact, with optional full candidate diagnostics."""
    if report is None or detailed:
        return report

    def compact(value):
        if isinstance(value, dict):
            return {
                key: compact(item)
                for key, item in value.items()
                if key
                not in {
                    "candidate_ranking",
                    "candidate_scores",
                    "positive_counts",
                    "mean_ranks",
                    "never_positive_features",
                }
            }
        if isinstance(value, list):
            return [compact(item) for item in value]
        return value

    return {
        key: compact(value) for key, value in report.items() if key not in {"candidate_ranking", "candidate_scores"}
    }


def make_model(train: pd.DataFrame, task: str, learner: str, parameters: dict, config: dict):
    """Construct matched preprocessing and a learner using no test information."""
    from featcopilot.utils.numeric_clipping import NumericQuantileClipper

    clip_quantiles = config.get("numeric_clip_quantiles")
    categorical_encoding = config.get("categorical_encoding", "one_hot")
    if categorical_encoding not in {"one_hot", "target"}:
        raise ValueError("categorical_encoding must be 'one_hot' or 'target'")
    if categorical_encoding == "target" and (task != "classification" or learner == "catboost"):
        raise ValueError("Target encoding supports classification with numeric/OHE learners, not native CatBoost")
    robust_steps = [] if clip_quantiles is None else [NumericQuantileClipper(quantiles=clip_quantiles)]
    numeric = train.select_dtypes(include=np.number).columns.tolist()
    if clip_quantiles is not None:
        numeric = [
            column
            for column in train
            if column in numeric
            or (
                pd.api.types.is_bool_dtype(train[column])
                and not (categorical_encoding == "target" and isinstance(train[column].dtype, pd.CategoricalDtype))
            )
        ]
    categorical = [col for col in train if col not in numeric]
    categorical_transformer: BaseEstimator
    if categorical_encoding == "target" and categorical:
        from featcopilot.utils.model_preprocessing import CrossFittedTargetEncoder

        categorical_transformer = CrossFittedTargetEncoder(random_state=config["seed"])
    else:
        categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    preprocessor = ColumnTransformer(
        [
            ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
            ("categorical", categorical_transformer, categorical),
        ],
        sparse_threshold=0,
    )
    if learner == "linear":
        model = (
            LogisticRegression(C=parameters["regularization"], max_iter=2000, random_state=config["seed"])
            if task == "classification"
            else Ridge(alpha=parameters["regularization"])
        )
    elif learner == "random_forest":
        cls = RandomForestClassifier if task == "classification" else RandomForestRegressor
        model = cls(
            n_estimators=100, max_depth=parameters["max_depth"], random_state=config["seed"], n_jobs=config["threads"]
        )
    elif learner == "lightgbm":
        from lightgbm import LGBMClassifier, LGBMRegressor

        cls = LGBMClassifier if task == "classification" else LGBMRegressor
        model = cls(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=parameters["num_leaves"],
            reg_lambda=1.0,
            verbosity=-1,
            deterministic=True,
            force_col_wise=True,
            random_state=config["seed"],
            n_jobs=config["threads"],
        )
    elif learner == "catboost":
        from catboost import CatBoostClassifier, CatBoostRegressor

        cls = CatBoostClassifier if task == "classification" else CatBoostRegressor
        model = cls(
            iterations=300,
            depth=parameters["depth"],
            learning_rate=0.05,
            l2_leaf_reg=3.0,
            random_seed=config["seed"],
            thread_count=config["threads"],
            task_type="CPU",
            cat_features=categorical,
            verbose=False,
            allow_writing_files=False,
        )
        return make_pipeline(
            *robust_steps, NativeCategoricalFrame(boolean_as_numeric=clip_quantiles is not None), model
        )
    elif learner == "xgboost":
        from xgboost import XGBClassifier, XGBRegressor

        cls = XGBClassifier if task == "classification" else XGBRegressor
        model = cls(
            n_estimators=300,
            max_depth=parameters["depth"],
            learning_rate=0.05,
            reg_lambda=1.0,
            tree_method="hist",
            device="cpu",
            random_state=config["seed"],
            n_jobs=config["threads"],
        )
    else:
        raise ValueError(f"Unknown learner {learner!r}")
    return make_pipeline(*robust_steps, preprocessor, model)


def regression_metrics(prediction: np.ndarray, y: pd.Series) -> dict[str, float]:
    """Score finite, row-aligned single-output regression predictions."""
    prediction = np.asarray(prediction)
    if prediction.shape != (len(y),) or prediction.dtype.kind not in "iuf" or not np.isfinite(prediction).all():
        raise ValueError("Regression predictions must be finite, numeric, one-dimensional and row-aligned")
    result = {
        "loss": float(np.sqrt(mean_squared_error(y, prediction))),
        "r2": float(r2_score(y, prediction)),
        "mae": float(mean_absolute_error(y, prediction)),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("Non-finite evaluation metric")
    return result


def metrics(model, X: pd.DataFrame, y: pd.Series, task: str) -> dict[str, float]:
    """Score without changing a model; lower ``loss`` is always better."""
    prediction = model.predict(X)
    if task == "classification":
        probability = model.predict_proba(X)
        classes = model.classes_
        result = {
            "loss": float(log_loss(y, probability, labels=classes)),
            "brier_sum": float(
                np.mean(
                    np.sum(
                        (np.asarray(probability, dtype=float) - (np.asarray(y)[:, None] == np.asarray(classes))) ** 2,
                        axis=1,
                    )
                )
            ),
            "accuracy": float(accuracy_score(y, prediction)),
            "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
            "macro_f1": float(f1_score(y, prediction, average="macro", zero_division=0)),
        }
        if len(classes) == 2:
            result["roc_auc"] = float(roc_auc_score(y, probability[:, 1], labels=classes))
            result["average_precision"] = float(average_precision_score(np.asarray(y) == classes[1], probability[:, 1]))
        else:
            result["roc_auc"] = float(roc_auc_score(y, probability, labels=classes, multi_class="ovr", average="macro"))
    else:
        result = regression_metrics(prediction, y)
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("Non-finite evaluation metric")
    return result


def target_preprocessing_diagnostics(model: Pipeline) -> dict:
    """Report fitted model-input widths without target maps or category vocabularies."""
    preprocessing = model.named_steps["columntransformer"]
    spans = preprocessing.output_indices_
    categorical = spans["categorical"]
    active = categorical.stop > categorical.start
    report = (
        dict(preprocessing.named_transformers_["categorical"].get_encoding_report())
        if active
        else {"requested": "target", "active": False}
    )
    report["total_encoded_width"] = max(span.stop for span in spans.values())
    report["categorical_encoded_width"] = categorical.stop - categorical.start
    return report


def run_case(payload: dict, workspace: Path, *, outer_predictions: dict[str, np.ndarray] | None = None) -> list[dict]:
    """Evaluate one outer fold; optionally capture final regression predictions in memory."""
    from threadpoolctl import threadpool_limits

    if "data_path" in payload:
        content = Path(payload["data_path"]).read_bytes()
        if digest(content) != payload["prepared_sha256"]:
            raise ValueError("Prepared dataset changed before worker execution")
        data = pickle.loads(content)
        payload = {**payload, **data}
    X, y = payload["X"], payload["y"]
    config, task, method = payload["config"], payload["task"], payload["method"]
    if outer_predictions is not None and task != "regression":
        raise ValueError("Outer prediction capture supports regression only")
    if method in TARGET_ENCODING_METHODS:
        if task != "classification":
            raise ValueError("Target encoding benchmark methods support classification only")
        if any(payload.get(name) is not None for name in ("groups", "timestamps", "label_end_times")):
            raise ValueError("Target encoding benchmark methods support IID only")
        config = {**config, "categorical_encoding": "target"}
    if method in ROBUST_METHODS:
        config = {**config, "numeric_clip_quantiles": ROBUST_CLIP_QUANTILES}
    if len(config["learners"]) > 1:
        rows = []
        for learner in config["learners"]:
            rows.extend(
                run_case(
                    {**payload, "config": {**config, "learners": [learner]}},
                    workspace,
                    outer_predictions=outer_predictions,
                )
            )
        return rows
    if method == "featcopilot-regression-average-robust":
        from benchmarks.regression_averaging import run_case as run_average_case

        if outer_predictions is not None:
            raise ValueError("Nested regression-average prediction capture is not supported")
        return run_average_case({**payload, "config": config}, workspace)
    if method in PLAN_VALIDATION_METHODS:
        from benchmarks.plan_validation import run_case as run_plan_case

        return run_plan_case({**payload, "config": config}, workspace)
    if method in PREDICTIVE_METHODS | {"llm-hybrid", "llm-hybrid-namespaced"}:
        config = {**config, "selection_learner": config["learners"][0]}
    config = {**config, **{name: payload.get(name) for name in ("groups", "timestamps", "label_end_times")}}
    config["column_descriptions"] = payload.get("column_descriptions", {})
    if (
        any(config[name] is not None for name in ("groups", "timestamps", "label_end_times"))
        and method not in {"baseline", "baseline-robust", "baseline-target-robust", "polynomial"} | PREDICTIVE_METHODS
    ):
        raise ValueError(f"{method} has no verified group/time-aware internal fitting protocol")
    train_idx, test_idx = payload["train_idx"], payload["test_idx"]
    base = {key: payload[key] for key in ("dataset", "source", "task", "method", "seed", "fold", "split_sha256")}
    base["family_id"] = payload.get("family_id", payload["dataset"])
    base["role"] = payload.get("role", "development")
    base["expected_cells"] = payload.get("expected_cells", 0)
    np.random.seed(config["seed"])
    random.seed(config["seed"])
    started = time.perf_counter()
    rows = []
    with threadpool_limits(limits=config["threads"]):
        if any(config[name] is not None for name in ("groups", "timestamps", "label_end_times")):
            from featcopilot.utils.splitting import make_holdout_splits

            fit_args = {
                name: config[name].iloc[train_idx].to_numpy() if config[name] is not None else None
                for name in ("groups", "timestamps", "label_end_times")
            }
            inner_a, inner_b = make_holdout_splits(
                y.iloc[train_idx].to_numpy(),
                task,
                1,
                config["seed"],
                test_size=0.25,
                groups=fit_args["groups"],
                timestamps=fit_args["timestamps"],
                label_end_times=fit_args["label_end_times"],
            )[0]
            inner_train, inner_valid = train_idx[inner_a], train_idx[inner_b]
        else:
            inner_train, inner_valid = train_test_split(
                train_idx,
                test_size=0.25,
                random_state=config["seed"],
                stratify=y.iloc[train_idx] if task == "classification" else None,
            )
        aligned = method in ALIGNED_METHODS
        inner_config = (
            {**config, "selection_model_parameters": candidate_parameters(config["learners"][0])[0]}
            if aligned
            else config
        )
        fe_start = time.perf_counter()
        inner_X, valid_X = generate_features(
            X.iloc[inner_train], X.iloc[inner_valid], y.iloc[inner_train], method, task, inner_config, workspace
        )
        inner_fe_seconds = time.perf_counter() - fe_start
        chosen: dict[str, dict] = {}
        tuning_seconds: dict[str, float] = {}
        chosen_inner_attrs: dict[str, dict] = {}
        pipeline_trials: dict[str, list[dict]] = {}
        for learner in config["learners"]:
            start = time.perf_counter()
            extra_fe_seconds = 0.0
            losses = []
            trial_attrs = []
            trials = []
            for parameter_index, parameters in enumerate(candidate_parameters(learner)):
                if aligned and parameter_index:
                    fe_start = time.perf_counter()
                    inner_X, valid_X = generate_features(
                        X.iloc[inner_train],
                        X.iloc[inner_valid],
                        y.iloc[inner_train],
                        method,
                        task,
                        {**config, "selection_model_parameters": parameters},
                        workspace,
                    )
                    extra_fe_seconds += time.perf_counter() - fe_start
                model = make_model(inner_X, task, learner, parameters, config)
                model.fit(inner_X.replace([np.inf, -np.inf], np.nan), y.iloc[inner_train])
                losses.append(
                    metrics(model, valid_X.replace([np.inf, -np.inf], np.nan), y.iloc[inner_valid], task)["loss"]
                )
                trial_attrs.append(dict(inner_X.attrs))
                if method in TARGET_ENCODING_METHODS:
                    trial_attrs[-1]["categorical_preprocessing"] = target_preprocessing_diagnostics(model)
                if aligned:
                    trials.append(
                        {
                            "parameters": parameters,
                            "loss": losses[-1],
                            "n_output_features": inner_X.shape[1],
                            "selection_report": selection_diagnostics(
                                inner_X.attrs.get("selection_report"), config.get("detailed_diagnostics", False)
                            ),
                            **(
                                {"categorical_preprocessing": trial_attrs[-1]["categorical_preprocessing"]}
                                if method in TARGET_ENCODING_METHODS
                                else {}
                            ),
                        }
                    )
            best = int(np.argmin(losses))
            chosen[learner] = {"parameters": candidate_parameters(learner)[best], "inner_losses": losses}
            chosen_inner_attrs[learner] = trial_attrs[best]
            pipeline_trials[learner] = trials
            tuning_seconds[learner] = time.perf_counter() - start - extra_fe_seconds
            inner_fe_seconds += extra_fe_seconds
        outer_config = (
            {**config, "selection_model_parameters": chosen[config["learners"][0]]["parameters"]} if aligned else config
        )
        fe_start = time.perf_counter()
        train_X, test_X = generate_features(
            X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], method, task, outer_config, workspace
        )
        outer_fe_seconds = time.perf_counter() - fe_start
        if not train_X.index.equals(X.iloc[train_idx].index) or not test_X.index.equals(X.iloc[test_idx].index):
            raise ValueError("Feature engineering changed row alignment")
        if not train_X.columns.equals(test_X.columns):
            raise ValueError("Feature engineering produced mismatched train/test schemas")
        for learner in config["learners"]:
            start = time.perf_counter()
            model = make_model(train_X, task, learner, chosen[learner]["parameters"], config)
            model.fit(train_X.replace([np.inf, -np.inf], np.nan), y.iloc[train_idx])
            fit_seconds = time.perf_counter() - start
            start = time.perf_counter()
            regression_details = {}
            if task == "regression":
                prediction = np.asarray(model.predict(test_X.replace([np.inf, -np.inf], np.nan)))
                scores = regression_metrics(prediction, y.iloc[test_idx])
                regression_details = {
                    "outer_prediction_sha256": digest(np.asarray(prediction, dtype="<f8").tobytes()),
                    "outer_feature_names_sha256": digest(json.dumps(list(train_X.columns)).encode()),
                }
                if outer_predictions is not None:
                    outer_predictions[learner] = prediction.copy()
            else:
                scores = metrics(model, test_X.replace([np.inf, -np.inf], np.nan), y.iloc[test_idx], task)
            predict_seconds = time.perf_counter() - start
            rows.append(
                {
                    **base,
                    "learner": learner,
                    "status": "success",
                    **scores,
                    **regression_details,
                    **chosen[learner],
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "n_input_features": X.shape[1],
                    "n_output_features": train_X.shape[1],
                    "inner_fe_seconds": inner_fe_seconds,
                    "outer_fe_seconds": outer_fe_seconds,
                    "tuning_seconds": tuning_seconds[learner],
                    "model_fit_seconds": fit_seconds,
                    "predict_seconds": predict_seconds,
                    "pipeline_seconds": inner_fe_seconds
                    + outer_fe_seconds
                    + tuning_seconds[learner]
                    + fit_seconds
                    + predict_seconds,
                    "case_seconds": time.perf_counter() - started,
                    "inner_selection_report": selection_diagnostics(
                        chosen_inner_attrs[learner].get("selection_report"), config.get("detailed_diagnostics", False)
                    ),
                    "outer_selection_report": selection_diagnostics(
                        train_X.attrs.get("selection_report"), config.get("detailed_diagnostics", False)
                    ),
                    "inner_llm_diagnostics": chosen_inner_attrs[learner].get("llm_diagnostics"),
                    "outer_llm_diagnostics": train_X.attrs.get("llm_diagnostics"),
                    "inner_llm_plan": chosen_inner_attrs[learner].get("llm_plan"),
                    "outer_llm_plan": train_X.attrs.get("llm_plan"),
                    "inner_reference_diagnostics": chosen_inner_attrs[learner].get("reference_diagnostics"),
                    "outer_reference_diagnostics": train_X.attrs.get("reference_diagnostics"),
                    **(
                        {
                            "selection_model_parameters": chosen[learner]["parameters"],
                            "inner_pipeline_trials": pipeline_trials[learner],
                        }
                        if aligned
                        else {}
                    ),
                    **(
                        {
                            "numeric_clip_quantiles": list(ROBUST_CLIP_QUANTILES),
                            "matched_baseline_method": "baseline-robust",
                        }
                        if method in ROBUST_METHODS
                        else {}
                    ),
                    **(
                        {
                            "categorical_encoding": "target",
                            "matched_encoding_baseline_method": "baseline-target-robust",
                            "inner_categorical_preprocessing": chosen_inner_attrs[learner]["categorical_preprocessing"],
                            "outer_categorical_preprocessing": target_preprocessing_diagnostics(model),
                        }
                        if method in TARGET_ENCODING_METHODS
                        else {}
                    ),
                }
            )
    return rows


def failure_rows(payload: dict, status: str, message: str, seconds: float) -> list[dict]:
    """Record every requested learner when a method/fold fails; never copy baseline scores."""
    base = {key: payload[key] for key in ("dataset", "source", "task", "method", "seed", "fold", "split_sha256")}
    base["family_id"] = payload.get("family_id", payload["dataset"])
    base["role"] = payload.get("role", "development")
    base["expected_cells"] = payload.get("expected_cells", 0)
    return [
        {**base, "learner": learner, "status": status, "error": message, "case_seconds": seconds}
        for learner in payload["config"]["learners"]
    ]


def stop_process_tree(process: subprocess.Popen) -> None:
    """Terminate only the worker PID and its descendants on a time limit."""
    import psutil

    try:
        parent = psutil.Process(process.pid)
        children = parent.children(recursive=True)
    except psutil.NoSuchProcess:
        children = []
    for child in reversed(children):
        try:
            child.kill()
        except psutil.NoSuchProcess:
            continue
    if process.poll() is None:
        process.kill()
    process.wait()
    psutil.wait_procs(children, timeout=5)


def run_isolated(
    payload: dict, directory: Path, timeout: float, *, worker_module: str = "benchmarks.tabular"
) -> list[dict]:
    """Enforce a wall-clock cap on the entire case, including feature engineering."""
    import psutil

    if worker_module not in {
        "benchmarks.tabular",
        "benchmarks.calibration_evaluation",
        "benchmarks.blending_evaluation",
        "benchmarks.regression_training_audit",
    }:
        raise ValueError("Unsupported benchmark worker module")
    seed = payload["seed"]
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or not 0 <= seed <= 4294967295:
        raise ValueError("Worker seed must be an integer between 0 and 4294967295")
    directory.mkdir(parents=True, exist_ok=False)
    input_path, output_path = directory / "input.pkl", directory / "result.json"
    input_path.write_bytes(pickle.dumps(payload))
    env = os.environ.copy()
    env.update({key: str(payload["config"]["threads"]) for key in THREAD_VARIABLES})
    # Python's hash seed must be fixed before interpreter startup, not inside the worker.
    env["PYTHONHASHSEED"] = str(int(seed))
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    start = time.perf_counter()
    peak_rss_mb = 0.0
    with (directory / "worker.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                worker_module,
                "--worker",
                str(input_path),
                "--worker-output",
                str(output_path),
            ],
            cwd=directory,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        status = None
        try:
            while process.poll() is None:
                try:
                    parent = psutil.Process(process.pid)
                    processes = [parent, *parent.children(recursive=True)]
                    rss = 0
                    for running in processes:
                        try:
                            rss += running.memory_info().rss
                        except psutil.NoSuchProcess:
                            continue
                    peak_rss_mb = max(peak_rss_mb, rss / 1024**2)
                except psutil.NoSuchProcess:
                    pass
                if time.perf_counter() - start >= timeout:
                    status = ("timeout", f"Case exceeded {timeout:.1f}s wall-clock limit")
                elif peak_rss_mb > payload["config"].get("memory_limit_mb", 4096):
                    status = ("memory_limit", "Worker tree exceeded the configured RSS memory limit")
                if status is not None:
                    stop_process_tree(process)
                    break
                try:
                    process.wait(timeout=min(0.2, max(0.001, timeout - (time.perf_counter() - start))))
                except subprocess.TimeoutExpired:
                    continue
        except KeyboardInterrupt:
            stop_process_tree(process)
            input_path.unlink(missing_ok=True)
            raise
        if status is not None:
            rows = failure_rows(payload, status[0], status[1], time.perf_counter() - start)
        else:
            if process.returncode == 0 and output_path.exists():
                rows = json.loads(output_path.read_text(encoding="utf-8"))
            else:
                rows = failure_rows(
                    payload, "error", f"Worker exited {process.returncode}; see worker.log", time.perf_counter() - start
                )
    input_path.unlink()  # Do not retain copies of public datasets in each case.
    for row in rows:
        row["worker_python_hash_seed"] = int(seed)
        row["worker_wall_seconds"] = time.perf_counter() - start
        row["peak_tree_rss_mb"] = peak_rss_mb
        row["log"] = directory.name + "/worker.log"
    return rows


def summarize(rows: list[dict], folds: int, n_seeds: int, *, baseline_method: str = "baseline") -> list[dict]:
    """Aggregate paired dataset means, treating datasets rather than CV folds as units."""
    frame = pd.DataFrame(rows)
    if frame.empty:
        return []
    if "role" not in frame:
        frame["role"] = "development"
    if "family_id" not in frame:
        frame["family_id"] = frame["dataset"]
    frame["family_id"] = frame["family_id"].fillna(frame["dataset"])
    summary = []
    for (source, role, task, learner), group in frame.groupby(["source", "role", "task", "learner"], sort=True):
        datasets = sorted(group["dataset"].unique())
        baseline = group[(group.method == baseline_method) & (group.status == "success")]
        for method, subset in group.groupby("method", sort=True):
            completed: dict[str, tuple[float, float]] = {}
            for dataset in datasets:
                base = baseline[baseline.dataset == dataset]
                other = subset[(subset.dataset == dataset) & (subset.status == "success")]
                keys = ["dataset", "seed", "fold", "split_sha256"]
                paired = base.merge(other, on=keys, suffixes=("_baseline", "_method"), validate="one_to_one")
                expected = int(other["expected_cells"].max()) if len(other) and "expected_cells" in other else 0
                if len(paired) != (expected or folds * n_seeds):
                    continue
                base_loss = float(paired.loss_baseline.mean())
                if base_loss <= 0:
                    continue
                completed[dataset] = (
                    1.0 - float(paired.loss_method.mean()) / base_loss,
                    float(paired.pipeline_seconds_method.mean()),
                )
            reductions, times = [], []
            for _, family in group.groupby("family_id", sort=True):
                names = set(family["dataset"])
                if not names.issubset(completed):
                    continue
                reductions.append(float(np.mean([completed[name][0] for name in names])))
                times.append(float(np.mean([completed[name][1] for name in names])))
            values = np.asarray(reductions)
            if len(values) >= 2:
                rng = np.random.default_rng(2026)
                means = rng.choice(values, size=(5000, len(values)), replace=True).mean(axis=1)
                ci = np.quantile(means, [0.025, 0.975]).tolist()
            else:
                ci = None
            summary.append(
                {
                    "source": source,
                    "role": role,
                    "task": task,
                    "learner": learner,
                    "method": method,
                    "reference_method": baseline_method,
                    "requested_datasets": len(datasets),
                    "complete_paired_datasets": len(completed),
                    "requested_families": int(group.family_id.nunique()),
                    "complete_paired_families": len(values),
                    "successful_folds": int((subset.status == "success").sum()),
                    "requested_folds": len(subset),
                    "wins": int((values > 0.01).sum()),
                    "ties": int((np.abs(values) <= 0.01).sum()),
                    "losses": int((values < -0.01).sum()),
                    "mean_loss_reduction": float(values.mean()) if len(values) else None,
                    "median_loss_reduction": float(np.median(values)) if len(values) else None,
                    "dataset_bootstrap_95_ci": ci,
                    "mean_pipeline_seconds": float(np.mean(times)) if times else None,
                }
            )
    return summary


def write_report(output: Path, manifest: dict, rows: list[dict]) -> None:
    """Persist raw evidence and descriptive, coverage-qualified summaries."""
    requested_methods = set(manifest["config"].get("methods", [row["method"] for row in rows]))
    reference_method = next(
        (
            method
            for method in ("baseline", "baseline-robust", "baseline-discovery-robust")
            if method in requested_methods
        ),
        "baseline",
    )
    summary = summarize(
        rows, manifest["config"]["folds"], len(manifest["config"]["seeds"]), baseline_method=reference_method
    )
    robust_rows = (
        [row for row in rows if row["method"] in ROBUST_METHODS - PLAN_VALIDATION_METHODS]
        if "baseline-robust" in requested_methods
        else []
    )
    robust_summary = (
        summarize(
            robust_rows,
            manifest["config"]["folds"],
            len(manifest["config"]["seeds"]),
            baseline_method="baseline-robust",
        )
        if robust_rows
        else []
    )
    plan_rows = [row for row in rows if row["method"] in PLAN_VALIDATION_METHODS]
    plan_summary = (
        summarize(
            plan_rows,
            manifest["config"]["folds"],
            len(manifest["config"]["seeds"]),
            baseline_method="baseline-discovery-robust",
        )
        if plan_rows
        else []
    )
    target_rows = [row for row in rows if row["method"] in TARGET_ENCODING_METHODS]
    target_summary = (
        summarize(
            target_rows,
            manifest["config"]["folds"],
            len(manifest["config"]["seeds"]),
            baseline_method="baseline-target-robust",
        )
        if target_rows
        else []
    )
    write_json(
        output / "results.json",
        {
            "manifest": manifest,
            "rows": rows,
            "summary": summary,
            **({"matched_robust_summary": robust_summary} if robust_rows else {}),
            **({"matched_plan_summary": plan_summary} if plan_rows else {}),
            **({"matched_target_summary": target_summary} if target_rows else {}),
        },
    )
    lines = [
        "# Tabular development benchmark",
        "",
        "**Benchmark measurement, not a SOTA claim.** Dataset identity, admission roles and split policies",
        "are recorded in the manifest. Legacy manifests are not automatically rights-reviewed or confirmation data.",
        "",
        "Primary loss: classification log loss; regression RMSE (lower is better).",
        "Learners are tuned separately per method on inner validation, never selected using outer-test scores.",
        "Synthetic diagnostics are reported separately. All requested failures count toward coverage.",
        "",
        "OpenFE-stateless is an explicitly restricted variant, not full OpenFE. AutoFeat uses numeric generation.",
        "OpenFE-inductive searches all supported operator families but freezes training statistics at inference;",
        "its native variant retains all selected additions, while the width-matched variant retains the top-ranked budget.",
        "AutoFeat-default uses native generation/selection defaults and output width with train-fitted categorical preprocessing.",
        "All methods retain original categorical features for identical downstream one-hot preprocessing.",
        "FeatCopilot-task changes only explicit task routing; FeatCopilot-predictive uses learner-aligned holdout selection.",
        "FeatCopilot-paired is an opt-in development variant using paired-gain uncertainty instead of absolute-loss uncertainty.",
        "Extended variants add balanced numeric pairs and frozen training-observed categorical pair indicators; defaults remain standard.",
        "Importance variants fit a training-only joint learner after exact candidate deduplication; holdout subset validation remains independent.",
        "Frequency-paired adds bounded frozen unary-frequency candidates to importance-paired; mappings never learn from validation or test rows.",
        "Frequency-aligned separately fits each declared learner/selection setting during inner tuning and refits the winning configuration.",
        "Regression-average-robust averages independently tuned raw/FE predictions at fixed 0.5 weights; it retains separate component plans, not a feature union, and charges both fits.",
        "Robust variants cap numeric model inputs to their fitted training minimum/maximum; incremental FE value is reported against baseline-robust separately.",
        "Bagged-aligned-robust averages five training-only candidate-importance rankings before subset validation and during final ranking; it makes no formal stability-selection error-control claim.",
        "Plan-gated-robust reserves a discovery-independent gate, tests one frozen proposal, and refits only unsupervised mapping state after its decision.",
        "Its matched baseline-discovery-robust uses the same discovery-only raw-model tuning; it is not interchangeable with the ordinary robust baseline.",
        "Target-encoding variants cross-fit categorical model inputs inside each training partition; their incremental FE comparison uses baseline-target-robust.",
        "Unselected and selection-only variants isolate the legacy stages; unselected output retains the generator's full planned width.",
        "Other predictive variants fit a separate representation per learner with its predeclared middle search candidate.",
        "Each protocol-v3 worker has one learner and an independent budget. Native CatBoost uses categorical data;",
        "its internal predictive-selection estimator uses numeric preprocessing, a disclosed representation difference.",
        "",
        "## Configuration",
        "",
        "```json",
        json.dumps(manifest["config"], indent=2),
        "```",
        "",
        "## Paired dataset-level results",
        "",
        f"Reference method: `{reference_method}`, determined by the declared methods, not observed success or scores.",
        "Loss reduction is 1 - mean(method loss) / mean(baseline loss), first averaged within each dataset.",
        "W/T/L uses a predeclared 1% relative-loss tolerance, not a significance test.",
        "95% intervals bootstrap origin families (not dependent folds), are descriptive and not multiplicity-adjusted.",
        "Rows with different complete-pair coverage must not be ranked against one another.",
        "",
        "| Source / role | Task | Learner | Method | Paired datasets / families | Cells OK/all | W/T/L | Mean loss reduction | 95% CI |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    if manifest.get("source_changed", False):
        lines.insert(2, "**INVALID COMPARISON: source changed during measurement. Rerun with frozen code.**\n")

    def append_summary(items: list[dict]) -> None:
        for item in items:
            reduction = "n/a" if item["mean_loss_reduction"] is None else f"{item['mean_loss_reduction']:+.2%}"
            interval = item["dataset_bootstrap_95_ci"]
            ci = "n/a" if interval is None else f"[{interval[0]:+.2%}, {interval[1]:+.2%}]"
            lines.append(
                f"| {item['source']} / {item['role']} | {item['task']} | {item['learner']} | {item['method']} | "
                f"{item['complete_paired_datasets']}/{item['requested_datasets']} / "
                f"{item['complete_paired_families']}/{item['requested_families']} | "
                f"{item['successful_folds']}/{item['requested_folds']} | {item['wins']}/{item['ties']}/{item['losses']} | {reduction} | {ci} |"
            )

    append_summary(summary)
    if robust_rows and reference_method != "baseline-robust":
        lines.extend(
            [
                "",
                "## Incremental feature value against the matched robust raw learner",
                "",
                "These reductions use baseline-robust on the same complete splits, not the unclipped raw baseline above.",
                "Clipping gains alone are not counted as feature-engineering gains. Missing robust controls remain unpaired.",
                "",
                "| Source / role | Task | Learner | Method | Paired datasets / families | Cells OK/all | W/T/L | Mean loss reduction | 95% CI |",
                "|---|---|---|---|---|---|---|---|---|",
            ]
        )
        append_summary(robust_summary)
    if plan_rows and reference_method != "baseline-discovery-robust":
        lines.extend(
            [
                "",
                "## Fixed-plan method against discovery-matched raw tuning",
                "",
                "These reductions use baseline-discovery-robust with identical discovery-only raw hyperparameter tuning.",
                "Both methods reserve the same gate rows from discovery; only the feature method uses them to accept or reject one plan.",
                "",
                "| Source / role | Task | Learner | Method | Paired datasets / families | Cells OK/all | W/T/L | Mean loss reduction | 95% CI |",
                "|---|---|---|---|---|---|---|---|---|",
            ]
        )
        append_summary(plan_summary)
    if target_rows:
        lines.extend(
            [
                "",
                "## FE versus identically target-encoded raw inputs",
                "",
                "Reference: baseline-target-robust; encoder changes alone are not counted as FE improvement.",
                "Target encoding affects only model preprocessing, not raw feature-plan width or candidate operands.",
                "",
                "| Source / role | Task | Learner | Method | Paired datasets / families | Cells OK/all | W/T/L | Mean loss reduction | 95% CI |",
                "|---|---|---|---|---|---|---|---|---|",
            ]
        )
        append_summary(target_summary)
    lines.extend(
        [
            "",
            "## Per-dataset means (complete folds only)",
            "",
            "| Dataset | Learner | Method | Primary loss | FE seconds (inner + outer) | Pipeline seconds | Features |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    if rows:
        frame = pd.DataFrame(rows)
        for (dataset, learner, method), group in frame.groupby(["dataset", "learner", "method"], sort=True):
            expected = int(group["expected_cells"].max()) if "expected_cells" in group else 0
            if (group.status == "success").all() and len(group) == (
                expected or manifest["config"]["folds"] * len(manifest["config"]["seeds"])
            ):
                lines.append(
                    f"| {dataset} | {learner} | {method} | {group.loss.mean():.6f} | "
                    f"{(group.inner_fe_seconds + group.outer_fe_seconds).mean():.2f} | "
                    f"{group.pipeline_seconds.mean():.2f} | {group.n_output_features.mean():.1f} |"
                )
    lines.extend(["", "## Failures / timeouts", ""])
    for row in rows:
        if row["status"] != "success":
            message = row["error"].replace("\n", " ")[:300]
            lines.append(
                f"- {row['dataset']} / {row['seed']} / fold {row['fold']} / {row['method']} / {row['learner']}: {row['status']}: {message}"
            )
    if all(row["status"] == "success" for row in rows):
        lines.append("None.")
    lines.extend(
        [
            "",
            "Raw cell metrics and provenance: local `results.json`. Completed runs also write `archive.json`,",
            "which lists the authoritative lossless gzip parts and their uncompressed checksum.",
            "",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def source_files() -> list[Path]:
    """List source inputs without generated results, prepared data or caches."""
    paths = sorted((ROOT / "featcopilot").rglob("*.py")) + sorted((ROOT / "benchmarks").rglob("*.py"))
    paths = [
        path
        for path in paths
        if not {"tabular-results", ".cache", "__pycache__"}.intersection(path.relative_to(ROOT).parts)
    ]
    paths += [ROOT / "pyproject.toml", SUITE_PATH]
    families = ROOT / "benchmarks" / "families.json"
    if families.exists():
        paths.append(families)
    paths.extend(sorted((ROOT / "benchmarks" / "suites").glob("*.json")))
    return paths


def source_fingerprint() -> str:
    """Hash current code and manifests, including dirty working-tree changes."""
    paths = source_files()
    contents = b"".join(str(path.relative_to(ROOT)).encode() + b"\0" + path.read_bytes() for path in paths)
    return digest(contents)


def main(argv: list[str] | None = None) -> int:
    """Run a bounded benchmark or an isolated case worker."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks" / "tabular-results")
    parser.add_argument(
        "--suite", type=Path, nargs="+", help="Validated dataset manifest(s); default: legacy pinned suite"
    )
    parser.add_argument("--datasets", nargs="+", help="Dataset names from the selected manifest")
    parser.add_argument("--methods", nargs="+", choices=METHODS)
    parser.add_argument("--learners", nargs="+", choices=LEARNERS)
    parser.add_argument("--folds", type=int)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--max-rows", type=int, help="Fixed uniform subsample; 0 uses all rows")
    parser.add_argument("--max-added-features", type=int)
    parser.add_argument("--max-candidates", type=int, help="Predictive strategy candidate budget")
    parser.add_argument("--stability-repeats", type=int)
    parser.add_argument("--screening", choices=["auto", "linear", "tree", "importance"])
    parser.add_argument("--threads", type=int)
    parser.add_argument(
        "--memory-limit-mb",
        type=int,
        help="Worker-tree RSS cap; shared pages may be counted more than once",
    )
    parser.add_argument("--case-timeout", type=float, help="Whole pipeline limit per dataset/split/method/learner cell")
    parser.add_argument(
        "--total-seconds", type=float, help="Cumulative active execution budget, including prior resumed work"
    )
    parser.add_argument(
        "--synthetic",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Also run independent synthetic diagnostics",
    )
    parser.add_argument("--allow-confirmation", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--detailed-diagnostics", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--live-llm",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Explicitly approve requested LLM methods",
    )
    parser.add_argument("--llm-ledger", type=Path, help="Shared durable 100-attempt allowance across runs/retries")
    parser.add_argument(
        "--allow-soft-llm-tokens",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Explicitly accept best-effort, not hard, 4096 output tokens",
    )
    parser.add_argument("--resume", action="store_true", help="Resume only fingerprint-identical cells in --output")
    parser.add_argument(
        "--retry-failed", action="store_true", help="Retry failed cells with unchanged semantic configuration"
    )
    parser.add_argument(
        "--prepare-only", action="store_true", help="Validate/admit/download data without fitting models"
    )
    parser.add_argument("--list-datasets", action="store_true", help="Show manifest inventory without downloading")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.worker:
        # Only the parent creates these local, trusted task payloads.
        payload = pickle.loads(args.worker.read_bytes())
        start = time.perf_counter()
        try:
            rows = run_case(payload, args.worker.parent)
        except Exception as exc:
            logger.exception("Benchmark case failed")
            rows = failure_rows(payload, "error", f"{type(exc).__name__}: {exc}", time.perf_counter() - start)
        write_json(args.worker_output, rows)
        return 0
    from benchmarks.execution import execute

    try:
        return execute(args)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        logger.error("%s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
