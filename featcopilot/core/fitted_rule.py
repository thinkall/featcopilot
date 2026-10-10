"""Validated, JSON-serializable specifications for bounded fitted operators."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

RuleOperator = Literal[
    "standardize", "clip", "frequency", "group_mean", "target_mean", "rare_category", "multiply", "divide", "log1p"
]


class _Parameters(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class _ClipParameters(_Parameters):
    lower_quantile: float = Field(default=0.01, ge=0, le=1)
    upper_quantile: float = Field(default=0.99, ge=0, le=1)

    @model_validator(mode="after")
    def _ordered_bounds(self) -> "_ClipParameters":
        if self.lower_quantile > self.upper_quantile:
            raise ValueError("lower_quantile must not exceed upper_quantile")
        return self


class _TargetParameters(_Parameters):
    smoothing: float = Field(default=10.0, ge=0)


class _RareParameters(_Parameters):
    min_frequency: float = Field(default=0.01, ge=0, le=1)
    other_label: str = Field(default="__other__", min_length=1)


_PARAMETERS: dict[str, type[_Parameters]] = {
    "standardize": _Parameters,
    "clip": _ClipParameters,
    "frequency": _Parameters,
    "group_mean": _Parameters,
    "target_mean": _TargetParameters,
    "rare_category": _RareParameters,
    "multiply": _Parameters,
    "divide": _Parameters,
    "log1p": _Parameters,
}


class FittedRuleSpec(BaseModel):
    """
    Specify one allowlisted feature without executable code.

    Parameters
    ----------
    name : str
        Unique output name, distinct from every input column.
    operator : str
        One of standardize, clip, frequency, group_mean, target_mean,
        rare_category, multiply, divide, or log1p.
    columns : list[str]
        One source column, except group_mean which takes [group, numeric_value].
        Multiply takes two distinct numeric columns; divide takes distinct
        [numerator, denominator] columns. Log1p takes one numeric column.
        Sources must be original input columns, not other rule outputs.
    parameters : dict, optional
        Operator-specific options. Clip accepts lower_quantile/upper_quantile
        (defaults 0.01/0.99); target_mean accepts nonnegative smoothing
        (default 10); rare_category accepts min_frequency (fraction, default
        0.01) and other_label (default "__other__"). Other operators take none.
    explanation : str, optional
        Human-readable purpose, retained with the fitted feature metadata.

    Notes
    -----
    Missing/nonfinite numerical values produce zero for standardize and clip.
    Standardization uses population standard deviation, unit scale for
    constant/all-missing training data, and saturates overflow to finite
    float64 limits. All-missing clipping bounds default to zero.
    Frequency uses training row fractions; missing and unknown categories
    produce zero. Group and target means use the training global mean for
    missing/unknown categories (zero if no valid group-mean values exist).
    Cross-fitted target means use only the fold's eligible training labels;
    the initial temporal prefix without a validation fold receives a fixed
    zero prior rather than any full-training target statistic.
    Numeric/boolean binary targets retain fixed 0/1 codes even if only one
    class is observed. Other binary labels use sorted class codes stored with
    the fitted state; comparisons of those codes assume a fixed label vocabulary.
    Rare-category rules require strings and map missing/unknown/rare values
    to other_label. Category keys distinguish strings, numbers and booleans.
    Multiply, divide and log1p are rowwise and learn no statistics. Log1p means
    log(1 + abs(x)), matching TabularEngine. Missing/nonfinite operands and
    zero denominators produce NaN; finite-input overflow saturates to float64
    limits. These operators accept no parameters.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)

    name: str = Field(min_length=1)
    operator: RuleOperator
    columns: list[str] = Field(min_length=1, max_length=2)
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    explanation: str = ""

    @model_validator(mode="after")
    def _validate_rule(self) -> "FittedRuleSpec":
        if not self.name.strip() or any(not column.strip() for column in self.columns):
            raise ValueError("Rule name and columns must not be blank")
        expected = 2 if self.operator in ("group_mean", "multiply", "divide") else 1
        if len(self.columns) != expected or len(set(self.columns)) != len(self.columns):
            raise ValueError(f"{self.operator} requires {expected} distinct source column(s)")
        validated = _PARAMETERS[self.operator].model_validate(self.parameters)
        object.__setattr__(self, "parameters", validated.model_dump(mode="json"))
        return self


class FittedRulePlan(BaseModel):
    """A bounded provider response containing only typed feature rules."""

    model_config = ConfigDict(extra="forbid", strict=True)

    rules: list[FittedRuleSpec] = Field(max_length=100)

    @model_validator(mode="after")
    def _unique_names(self) -> "FittedRulePlan":
        names = [rule.name for rule in self.rules]
        if len(names) != len(set(names)):
            raise ValueError("Rule output names must be unique")
        return self
