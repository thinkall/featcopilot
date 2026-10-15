"""Optional, inductive competitive references (no optional imports at module load).

Validated dependency contract: OpenFE 0.0.12, AutoFeat 2.1.1, pandas 2.3.3,
scikit-learn 1.7.2, numpy 2.4.4. Install these in a benchmark-only environment,
along with OpenFE's LightGBM/pyarrow dependencies; do not add them to base deps.
The loaders require the exact reference releases, pandas >=2.2,<3 and sklearn
>=1.6,<1.8 (AutoFeat 2.1.1 uses the removed ``force_all_finite`` keyword).

Input contract: unique string column names, identical ordered train/test schemas,
unique training/label indexes with identical label sets, and finite nonmissing y.
Numeric input is median-imputed from train (all-missing columns use explicit zero);
categorical scalars may be strings, numbers, booleans or missing. Internal safe
names avoid upstream formula parsers. Original columns/values/indexes are returned
unchanged. Derived OpenFE columns may contain NaN/inf; downstream preprocessing
must handle these. No test labels are accepted and no test statistics are fitted.

OpenFE searches the full published order=1 numeric AND categorical candidate set,
using its default selection algorithm and an explicit train-internal 80/20 split.
Stock search evaluates candidate statistics within the supplied training frame;
this is NOT a cross-fitted-statistics variant of its search. Selected DAGs are
refitted on the full supplied train and applied independently to each batch,
NEVER with stock ``openfe.transform``. See OPERATOR_POLICY for the deliberately
inductive differences from stock transductive evaluation.

Both AutoFeat variants retain native returned features with all default transformations,
two engineering steps and five selection runs. Its categorical_cols helper refits
OneHotEncoder on every transform in 2.1.1, so equivalent dense one-hot columns are
instead fitted here on train, with unseen categories all-zero. Raw categoricals
are also retained. No selected output is alphabetically truncated. Width caps
are rejected for these unranked references. ``autofeat-default`` preserves the
archived max_gb=1 generation-size estimate/subsampling control (not an upstream
default or hard memory bound). The separate, opt-in ``autofeat-native`` uses the
upstream max_gb=None default, disabling only generation-size-driven subsampling;
native selection-run resampling remains unchanged. Neither variant drops constant
columns or repairs upstream internal NaNs. External worker memory limits and
deadlines remain the caller's responsibility; neither variant guarantees a fit.

``autofeat-native-safe-scale-v1`` is a separate benchmark-only compatibility
reference; the parent registry supplies its ``-robust`` downstream wrapper.
It retains max_gb=None and the native generation/selection parameters. Only
feature/shuffled-noise scaling uses denominator=1 for wholly finite, actually
constant columns whose upstream std is exactly zero. It does not drop columns,
repair nonconstant std-underflow, impute internal NaNs, or change target scaling.
The private copy verifies AutoFeat 2.1.1 source hashes and exact numerical runtime
versions before fitting; its additional pins are recorded in search.compatibility.
Native references and prior failed results are not replaced.

Diagnostics have exactly these keys: status, method, task, seed, threads,
dependency_versions, input_features, input_columns, candidate_count,
selected_count, added_features, max_added_features, width_policy, feature_names,
feature_definitions, preprocessing, operator_policy, search, resources, seed_policy.
Counts describe candidate search (None for AutoFeat), native selected/returned
additions before a width cap, and actual appended columns, respectively.
feature_definitions is an ordered list of {name, expression}; input_columns maps
internal symbols to original columns. preprocessing.one_hot_levels maps AutoFeat
one-hot symbols to typed category tokens (or the distinct ``missing:`` sentinel).
method identifies the variant; resources.max_gb is 1 for autofeat-default and
None for autofeat-native. resources.hard_memory_limit=False describes this
adapter, not any external worker limit imposed by the caller.
For the compatibility reference only, search.compatibility records the policy,
dependency versions, upstream and compatibility-module source SHA-256 hashes.
Complex-valued inputs are rejected. Errors are raised, never baseline fallbacks.
ReferenceInputError/ReferenceWidthError signal invalid/unsupported metadata,
ReferenceDependencyError unavailable/incompatible libraries, ReferenceOperatorError
unsupported/malformed DAGs, ReferenceOutputError invalid upstream output.
Unmodified upstream fitting errors propagate to the caller's failure recorder.
"""

from __future__ import annotations

import hashlib
import json
import random
import tempfile
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype

_UNARY = {"abs", "log", "sqrt", "square", "sigmoid", "round", "residual", "freq"}
_BINARY = {"min", "max", "+", "-", "*", "/"}
_AGGREGATIONS = {
    "GroupByThenMin": "min",
    "GroupByThenMax": "max",
    "GroupByThenMean": "mean",
    "GroupByThenMedian": "median",
    "GroupByThenStd": "std",
    "GroupByThenNUnique": "nunique",
}
OPENFE_OPERATORS = (
    _UNARY | _BINARY | set(_AGGREGATIONS) | {"GroupByThenRank", "GroupByThenFreq", "Combine", "CombineThenFreq"}
)
OPERATOR_POLICY = {
    "statistics": "fit on supplied train only; frozen for all inference batches",
    "frequency": "absolute training counts; unseen=0; missing=NaN",
    "group_aggregates": "training aggregates; std ddof=1; unseen/missing group=NaN; nunique excludes missing",
    "group_rank": "training average percentile rank for ties; unseen values use right empirical CDF; missing/group unseen=NaN",
    "combine": "typed collision-free pairs; first-seen training codes; missing=-1; unseen=-2; fixed categorical dtype",
    "typed_keys": "scalar type is significant, unlike stock string-concatenated pairs or mixed-type group equality",
    "group_frequency": "training pair counts, aligned to original rows rather than pandas groupby.apply MultiIndex",
}


class ReferenceInputError(ValueError):
    """Invalid input schema, labels, task, or execution metadata."""


class ReferenceWidthError(ReferenceInputError):
    """Requested width policy is unsupported by this native reference."""


class ReferenceDependencyError(ImportError):
    """An optional reference dependency is missing or incompatible."""


class ReferenceOperatorError(NotImplementedError):
    """An OpenFE DAG contains an unsupported or malformed operator."""


class ReferenceOutputError(RuntimeError):
    """A reference library returned an invalid feature schema."""


def _key(value: Any) -> str | None:
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, float) and value == 0:
        value = 0.0
    if type(value) not in (str, int, float, bool):
        raise ReferenceInputError(f"Unsupported categorical scalar type: {type(value).__name__}")
    return json.dumps([type(value).__name__, value], ensure_ascii=True, separators=(",", ":"))


def _keys(series: pd.Series) -> list[str | None]:
    return [_key(value) for value in series]


def _pairs(left: pd.Series, right: pd.Series) -> list[str | None]:
    return [
        None if a is None or b is None else json.dumps([a, b], separators=(",", ":"))
        for a, b in zip(_keys(left), _keys(right), strict=True)
    ]


def _names(prefix: str, expressions: Sequence[str], originals: Sequence[str]) -> list[str]:
    occupied = set(originals)
    result = []
    for expression in expressions:
        base = f"reference_{prefix}_{hashlib.sha256(expression.encode()).hexdigest()[:16]}"
        name, suffix = base, 1
        while name in occupied:
            name = f"{base}_{suffix}"
            suffix += 1
        occupied.add(name)
        result.append(name)
    return result


@dataclass(frozen=True)
class _Spec:
    name: str
    children: tuple[_Spec, ...] | None = None

    @property
    def expression(self) -> str:
        if self.children is None:
            return json.dumps(self.name)
        return f"{self.name}({','.join(child.expression for child in self.children)})"


def _compile(node: Any, visiting: set[int]) -> _Spec:
    if id(node) in visiting or not isinstance(getattr(node, "name", None), str):
        raise ReferenceOperatorError("OpenFE nodes must be acyclic and have string names")
    if not hasattr(node, "children"):
        return _Spec(node.name)
    if node.name not in OPENFE_OPERATORS:
        raise ReferenceOperatorError(f"Unsupported OpenFE operator: {node.name!r}")
    arity = 1 if node.name in _UNARY else 2
    if not isinstance(node.children, (list, tuple)) or len(node.children) != arity:
        raise ReferenceOperatorError(f"OpenFE operator {node.name!r} requires {arity} children")
    visiting.add(id(node))
    children = tuple(_compile(child, visiting) for child in node.children)
    visiting.remove(id(node))
    return _Spec(node.name, children)


@dataclass
class _NodeState:
    lookup: dict[str, float] = field(default_factory=dict)
    ranks: dict[str, np.ndarray] = field(default_factory=dict)
    categories: tuple[float, ...] = ()


def _fit_state(op: str, values: list[pd.Series]) -> _NodeState:
    state = _NodeState()
    if op in {"freq", "Combine", "CombineThenFreq", "GroupByThenFreq"}:
        keys = _keys(values[0]) if op == "freq" else _pairs(*values)
        counts = Counter(key for key in keys if key is not None)
        if op == "Combine":
            state.lookup = {key: float(i) for i, key in enumerate(counts)}
            state.categories = (-2.0, -1.0, *state.lookup.values())
        else:
            state.lookup = {key: float(count) for key, count in counts.items()}
    elif op in _AGGREGATIONS or op == "GroupByThenRank":
        data = _keys(values[0]) if op == "GroupByThenNUnique" else values[0].to_numpy()
        groups = pd.DataFrame({"value": data, "group": _keys(values[1])}).groupby("group", sort=False)["value"]
        if op == "GroupByThenRank":
            state.ranks = {str(key): np.sort(group.dropna().to_numpy(dtype=float)) for key, group in groups}
        else:
            state.lookup = {str(key): float(value) for key, value in groups.agg(_AGGREGATIONS[op]).items()}
    return state


def _apply_operator(op: str, values: list[pd.Series], state: _NodeState) -> pd.Series:
    left = values[0]
    if op in {"freq", "Combine", "CombineThenFreq", "GroupByThenFreq"}:
        keys = _keys(left) if op == "freq" else _pairs(*values)
        if op == "Combine":
            codes = [-1.0 if key is None else state.lookup.get(key, -2.0) for key in keys]
            return pd.Series(pd.Categorical(codes, categories=state.categories), index=left.index)
        return pd.Series(
            [np.nan if key is None else state.lookup.get(key, 0.0) for key in keys], index=left.index, dtype=float
        )
    if op in _AGGREGATIONS:
        return pd.Series(
            [np.nan if key is None else state.lookup.get(key, np.nan) for key in _keys(values[1])],
            index=left.index,
            dtype=float,
        )
    if op == "GroupByThenRank":
        ranks = []
        for value, key in zip(left, _keys(values[1]), strict=True):
            distribution = state.ranks.get(key) if key is not None else None
            if pd.isna(value) or distribution is None or not len(distribution):
                ranks.append(np.nan)
            else:
                lo, hi = np.searchsorted(distribution, value, side="left"), np.searchsorted(
                    distribution, value, side="right"
                )
                ranks.append((lo + hi + 1) / (2 * len(distribution)) if lo < hi else hi / len(distribution))
        return pd.Series(ranks, index=left.index, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if op == "abs":
            result = left.abs()
        elif op == "log":
            result = np.log(left.replace(0, np.nan).abs())
        elif op == "sqrt":
            result = np.sqrt(left.abs())
        elif op == "square":
            result = np.square(left)
        elif op == "sigmoid":
            result = 1 / (1 + np.exp(-left))
        elif op == "round":
            result = np.floor(left)
        elif op == "residual":
            result = left - np.floor(left)
        elif op == "min":
            result = np.minimum(left, values[1])
        elif op == "max":
            result = np.maximum(left, values[1])
        elif op == "+":
            result = left + values[1]
        elif op == "-":
            result = left - values[1]
        elif op == "*":
            result = left * values[1]
        elif op == "/":
            result = left / values[1].replace(0, np.nan)
        else:
            raise ReferenceOperatorError(f"Unsupported OpenFE operator: {op!r}")
    return pd.Series(result, index=left.index, dtype=float)


class FittedOpenFEDAG:
    """Evaluate complete OpenFE node DAGs with immutable training statistics.

    Parameters
    ----------
    features : sequence
        OpenFE Node/FNode objects, including composed/shared subtrees. Native
        node caches are neither used nor mutated. This helper does not impute
        input; use generate_reference for the reference preprocessing contract.
    """

    def __init__(self, features: Sequence[Any]) -> None:
        self.specs_ = tuple(_compile(feature, set()) for feature in features)
        self.states_: dict[_Spec, _NodeState] = {}
        self.feature_names_: list[str] = []
        self._fitted = False

    def fit(self, train: pd.DataFrame) -> FittedOpenFEDAG:
        """Fit every selected subtree on train, reusing shared-node calculations."""
        self._fitted = False
        self.states_.clear()
        cache: dict[_Spec, pd.Series] = {}
        for spec in self.specs_:
            self._evaluate(spec, train, cache, fitting=True)
        self.feature_names_ = _names("openfe", self.expressions_, list(train.columns))
        self._fitted = True
        return self

    @property
    def expressions_(self) -> list[str]:
        """Return ordered structural expressions for the selected roots."""
        return [spec.expression for spec in self.specs_]

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Apply frozen state independently; preserve input row order and index."""
        if not self._fitted:
            raise RuntimeError("FittedOpenFEDAG must be fitted before transform")
        cache: dict[_Spec, pd.Series] = {}
        columns = {
            name: self._evaluate(spec, frame, cache, fitting=False)
            for name, spec in zip(self.feature_names_, self.specs_, strict=True)
        }
        return pd.DataFrame(columns, index=frame.index)

    def _evaluate(self, spec: _Spec, frame: pd.DataFrame, cache: dict[_Spec, pd.Series], *, fitting: bool) -> pd.Series:
        if spec not in cache:
            if spec.children is None:
                if spec.name not in frame:
                    raise ReferenceOperatorError(f"OpenFE leaf refers to missing column {spec.name!r}")
                cache[spec] = frame[spec.name]
            else:
                values = [self._evaluate(child, frame, cache, fitting=fitting) for child in spec.children]
                if fitting:
                    self.states_[spec] = _fit_state(spec.name, values)
                cache[spec] = _apply_operator(spec.name, values, self.states_[spec])
        return cache[spec]


class _PreparedInput:
    def __init__(self, train: pd.DataFrame) -> None:
        self.columns = list(train.columns)
        self.symbols = {f"x{i:06}": name for i, name in enumerate(self.columns)}
        self.numeric = [
            name
            for name in self.symbols
            if is_numeric_dtype(train[self.symbols[name]]) and not is_bool_dtype(train[self.symbols[name]])
        ]
        self.categorical = [name for name in self.symbols if name not in self.numeric]
        self.one_hot_levels: dict[str, str] = {}
        self.medians: dict[str, float] = {}
        self.all_missing: list[str] = []
        for symbol in self.numeric:
            series = train[self.symbols[symbol]].astype(float).replace([np.inf, -np.inf], np.nan)
            if series.notna().any():
                self.medians[symbol] = float(series.median())
            else:
                self.medians[symbol] = 0.0
                self.all_missing.append(symbol)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame(index=frame.index)
        for symbol, original in self.symbols.items():
            if symbol in self.medians:
                result[symbol] = (
                    pd.to_numeric(frame[original], errors="raise")
                    .astype(float)
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(self.medians[symbol])
                )
            else:
                result[symbol] = pd.Series(
                    [np.nan if key is None else key for key in _keys(frame[original])], index=frame.index, dtype=object
                )
        return result


def _require(package: str, release: str) -> None:
    try:
        installed = version(package)
        dependencies = {name: version(name) for name in ("pandas", "scikit-learn")}
    except PackageNotFoundError as error:
        raise ReferenceDependencyError(f"{package} reference dependency missing: {error}") from error
    if installed != release:
        raise ReferenceDependencyError(f"{package} reference requires {release}; found {installed}")
    for name, lower, upper in (("pandas", (2, 2), (3, 0)), ("scikit-learn", (1, 6), (1, 8))):
        parts = tuple(int(part) for part in dependencies[name].split(".")[:2])
        if not lower <= parts < upper:
            raise ReferenceDependencyError(
                f"{package} requires {name}>={'.'.join(map(str, lower))},"
                f"<{'.'.join(map(str, upper))}; found {dependencies[name]}"
            )


def _load_openfe() -> tuple[Callable[..., Any], Callable[..., Any]]:
    try:
        api = import_module("openfe")
        _require("openfe", "0.0.12")
        compatible = import_module("benchmarks.tabular_openfe")
    except ImportError as error:
        raise ReferenceDependencyError(f"openfe-inductive dependency unavailable: {error}") from error
    return compatible.CompatibleOpenFE, api.get_candidate_features


def _load_autofeat(task: str) -> Callable[..., Any]:
    try:
        api = import_module("autofeat")
        _require("autofeat", "2.1.1")
    except ImportError as error:
        raise ReferenceDependencyError(f"AutoFeat dependency unavailable: {error}") from error
    model: Callable[..., Any] = api.AutoFeatClassifier if task == "classification" else api.AutoFeatRegressor
    return model


@contextmanager
def _seeded(seed: int) -> Iterator[None]:
    python_state, numpy_state = random.getstate(), np.random.get_state()
    random.seed(seed)
    np.random.seed(seed)
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def _validate(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y: pd.Series,
    method: str,
    task: str,
    max_added_features: int | None,
    threads: int,
    seed: int,
) -> pd.Series:
    if method not in {"openfe-inductive", "autofeat-default", "autofeat-native", "autofeat-native-safe-scale-v1"}:
        raise ReferenceInputError(f"Unknown reference method {method!r}")
    if task not in {"classification", "regression"}:
        raise ReferenceInputError("Reference task must be explicitly classification or regression")
    if type(threads) is not int or threads < 1 or type(seed) is not int or not 0 <= seed < 2**32:
        raise ReferenceInputError("threads must be a positive integer; seed must be an integer in [0, 2**32)")
    if max_added_features is not None and (type(max_added_features) is not int or max_added_features < 0):
        raise ReferenceInputError("max_added_features must be None or a nonnegative integer")
    if (
        method in {"autofeat-default", "autofeat-native", "autofeat-native-safe-scale-v1"}
        and max_added_features is not None
    ):
        raise ReferenceWidthError(f"{method} requires native output (max_added_features=None); output is not ranked")
    if not isinstance(train, pd.DataFrame) or not isinstance(test, pd.DataFrame) or not isinstance(y, pd.Series):
        raise ReferenceInputError("Reference inputs must be DataFrame train/test and Series labels")
    if train.empty or not train.columns.is_unique or any(not isinstance(name, str) for name in train.columns):
        raise ReferenceInputError("Training input must be nonempty with unique string column names")
    if not train.columns.equals(test.columns):
        raise ReferenceInputError("Train/test columns must have identical names and order")
    if any(is_complex_dtype(dtype) for dtype in [*train.dtypes, *test.dtypes, y.dtype]):
        raise ReferenceInputError("Reference inputs and labels must not be complex-valued")
    if (
        not train.index.is_unique
        or not y.index.is_unique
        or len(train) != len(y)
        or not train.index.isin(y.index).all()
    ):
        raise ReferenceInputError("Training rows and labels require unique indexes with exactly matching label sets")
    labels = y.reindex(train.index)
    if labels.isna().any() or (is_numeric_dtype(labels) and not np.isfinite(labels.to_numpy(dtype=float)).all()):
        raise ReferenceInputError("Training labels must be nonmissing and finite")
    if task == "regression" and not is_numeric_dtype(labels):
        raise ReferenceInputError("Regression labels must be numeric")
    if task == "classification" and labels.nunique() < 2:
        raise ReferenceInputError("Classification labels must contain at least two classes")
    return labels


def _autofeat_inputs(
    prepared: _PreparedInput, train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not prepared.categorical:
        return train, test
    from sklearn.preprocessing import OneHotEncoder

    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    categories_train = train[prepared.categorical].fillna("missing:")
    encoded_train = encoder.fit_transform(categories_train)
    names = [
        f"cat_{symbol}_{i}"
        for symbol, categories in zip(prepared.categorical, encoder.categories_, strict=True)
        for i in range(len(categories))
    ]
    prepared.one_hot_levels = {
        f"cat_{symbol}_{i}": str(level)
        for symbol, categories in zip(prepared.categorical, encoder.categories_, strict=True)
        for i, level in enumerate(categories)
    }
    encoded_test = (
        encoder.transform(test[prepared.categorical].fillna("missing:")) if len(test) else np.empty((0, len(names)))
    )
    return (
        pd.concat([train[prepared.numeric], pd.DataFrame(encoded_train, index=train.index, columns=names)], axis=1),
        pd.concat([test[prepared.numeric], pd.DataFrame(encoded_test, index=test.index, columns=names)], axis=1),
    )


def _check_output(frame: Any, rows: int) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or len(frame) != rows or not frame.columns.is_unique:
        raise ReferenceOutputError("Reference output must be a DataFrame with matching row count and unique columns")
    if any(not isinstance(name, str) for name in frame.columns):
        raise ReferenceOutputError("Reference output columns must be strings")
    return frame


def generate_reference(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y: pd.Series,
    *,
    method: str,
    task: str,
    max_added_features: int | None,
    threads: int,
    seed: int,
    workspace: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Generate an explicitly labeled optional reference with no baseline fallback.

    Parameters
    ----------
    train, test : DataFrame
        Training and held-out covariates, with the module's mixed-input contract.
    y : Series
        Training labels, aligned by index rather than assumed positional order.
    method : {"openfe-inductive", "autofeat-default", "autofeat-native", "autofeat-native-safe-scale-v1"}
        Full-operator inductive OpenFE, archived AutoFeat with max_gb=1, or
        opt-in AutoFeat with upstream max_gb=None. Both AutoFeat variants retain
        default transformations/selection and rely on caller-enforced limits.
        The separately labeled safe-scale-v1 reference guards only verified
        constant feature/noise scaling with a pinned private fit/selector chain.
    task : {"classification", "regression"}
        Explicit supervised task; no target-cardinality inference.
    max_added_features : int or None
        OpenFE native-importance prefix length; None retains native output.
        AutoFeat only accepts None because its returned columns are not ranked.
    threads : int
        Positive worker/native-thread budget.
    seed : int
        Seed for upstream RNG use, restored on exit; see seed_policy diagnostics.
    workspace : Path
        Caller-owned temporary directory, used for a unique OpenFE feather file.

    Returns
    -------
    train_output, test_output : DataFrame
        Unchanged originals plus schema-identical derived columns.
    diagnostics : dict
        JSON-serializable fields listed in the module docstring.

    Raises
    ------
    ReferenceInputError, ReferenceDependencyError, ReferenceOperatorError, ReferenceOutputError
        Explicit metadata/dependency/operator/output failures. Upstream exceptions
        otherwise propagate, including search failures and resource exhaustion.
    """
    labels = _validate(train, test, y, method, task, max_added_features, threads, seed)
    prepared = _PreparedInput(train)
    prepared_train, prepared_test = prepared.transform(train), prepared.transform(test)
    max_gb = 1 if method == "autofeat-default" else None
    candidate_count: int | None = None
    search: dict[str, Any]
    from threadpoolctl import threadpool_limits

    with _seeded(seed), threadpool_limits(limits=threads):
        if method == "openfe-inductive":
            from sklearn.model_selection import train_test_split

            model, get_candidates = _load_openfe()
            candidates = get_candidates(
                numerical_features=prepared.numeric, categorical_features=prepared.categorical, order=1
            )
            FittedOpenFEDAG(candidates)  # Fail on unsupported operators, never filter them away.
            candidate_count = len(candidates)
            if not candidate_count:
                raise ReferenceOutputError("OpenFE published candidate generator returned no candidates")
            # Reset only internal indexes: upstream feather/groupby assume simple row IDs.
            fitting = prepared_train.reset_index(drop=True)
            target = pd.DataFrame({"target": labels.to_numpy()}, index=fitting.index)
            try:
                fit_index, val_index = train_test_split(
                    fitting.index,
                    test_size=0.2,
                    random_state=seed,
                    stratify=labels.to_numpy() if task == "classification" else None,
                )
            except ValueError as error:
                raise ReferenceInputError(
                    f"Cannot form OpenFE's train-internal 80/20 selection split: {error}"
                ) from error
            workspace.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                suffix=".feather", prefix="openfe-reference-", dir=workspace, delete=False
            ) as f:
                temporary = Path(f.name)
            try:
                selected = model().fit(
                    fitting,
                    target,
                    task=task,
                    train_index=fit_index,
                    val_index=val_index,
                    candidate_features_list=candidates,
                    categorical_features=prepared.categorical,
                    n_jobs=threads,
                    seed=seed,
                    tmp_save_path=str(temporary),
                    verbose=False,
                )
            finally:
                temporary.unlink(missing_ok=True)
            if not isinstance(selected, list):
                raise ReferenceOutputError("OpenFE fit must return its importance-ordered feature list")
            FittedOpenFEDAG(selected)
            selected_count = len(selected)
            dag = FittedOpenFEDAG(selected[:max_added_features]).fit(prepared_train)
            engineered_train, engineered_test = dag.transform(prepared_train), dag.transform(prepared_test)
            expressions = dag.expressions_
            width_policy = "native_output" if max_added_features is None else "native_importance_order"
            search = {
                "order": 1,
                "operator_filter": None,
                "selection": "upstream defaults",
                "validation_fraction": 0.2,
                "fit_rows": len(fit_index),
                "validation_rows": len(val_index),
            }
        else:
            compatibility: dict[str, Any] | None = None
            if method == "autofeat-native-safe-scale-v1":
                try:
                    compatible = import_module("benchmarks._autofeat_safe_scale")
                    compatibility = compatible.provenance()
                except ImportError as error:
                    raise ReferenceDependencyError(f"AutoFeat safe-scale-v1 dependency unavailable: {error}") from error
                model = (
                    compatible.SafeScaleAutoFeatClassifier
                    if task == "classification"
                    else compatible.SafeScaleAutoFeatRegressor
                )
            else:
                model = _load_autofeat(task)
            fitting, inference = _autofeat_inputs(prepared, prepared_train, prepared_test)
            engineer = model(max_gb=max_gb, n_jobs=threads, verbose=0)
            engineered_train = _check_output(engineer.fit_transform(fitting, labels.to_numpy()), len(train))
            engineered_test = _check_output(
                engineer.transform(inference) if len(test) else engineered_train.iloc[:0].copy(), len(test)
            )
            if not engineered_train.columns.equals(engineered_test.columns):
                raise ReferenceOutputError("AutoFeat transform schema differs from its fitted native output")
            expressions = [name for name in engineered_train if name not in prepared.numeric]
            engineered_train = engineered_train[expressions].set_axis(train.index)
            engineered_test = engineered_test[expressions].set_axis(test.index)
            selected_count = len(expressions)
            width_policy = "native_output"
            search = {
                "feateng_steps": 2,
                "featsel_runs": 5,
                "transformations": ["1/", "exp", "log", "abs", "sqrt", "^2", "^3"],
                "selection": "upstream defaults",
                "categorical_encoding": "adapter train-only one-hot",
            }
            if compatibility is not None:
                search["compatibility"] = compatibility
    names = _names(method.split("-")[0], expressions, list(train.columns))
    engineered_train.columns = engineered_test.columns = names
    package = "openfe" if method == "openfe-inductive" else "autofeat"
    dependencies = {}
    for dependency in (package, "numpy", "pandas", "scikit-learn"):
        try:
            dependencies[dependency] = version(dependency)
        except PackageNotFoundError as error:
            raise ReferenceDependencyError(f"Reference dependency metadata unavailable: {dependency}") from error
    diagnostics = {
        "status": "success",
        "method": method,
        "task": task,
        "seed": seed,
        "threads": threads,
        "dependency_versions": dependencies,
        "input_features": train.shape[1],
        "input_columns": prepared.symbols,
        "candidate_count": candidate_count,
        "selected_count": selected_count,
        "added_features": len(names),
        "max_added_features": max_added_features,
        "width_policy": width_policy,
        "feature_names": names,
        "feature_definitions": [
            {"name": name, "expression": expression} for name, expression in zip(names, expressions, strict=True)
        ],
        "preprocessing": {
            "numeric_medians": prepared.medians,
            "all_missing_numeric_zero": prepared.all_missing,
            "categorical_columns": prepared.categorical,
            "one_hot_levels": prepared.one_hot_levels,
            "categorical_missing": "preserved for OpenFE; train-fitted level for AutoFeat",
            "categorical_unknown": "OpenFE operator policy; AutoFeat one-hot all-zero",
        },
        "operator_policy": dict(OPERATOR_POLICY) if package == "openfe" else {},
        "search": search,
        "resources": {
            "max_gb": max_gb,
            "hard_memory_limit": False,
            "hard_deadline": "caller",
        },
        "seed_policy": "scoped numpy/Python seed restored on exit; upstream AutoFeat selection also seeds each run",
    }
    return pd.concat([train, engineered_train], axis=1), pd.concat([test, engineered_test], axis=1), diagnostics
