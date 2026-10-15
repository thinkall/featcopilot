"""Shared execution contract for generated feature code.

This is an in-process execution helper, not a security sandbox.
"""

import ast
import builtins
from typing import Any

import numpy as np
import pandas as pd

_BUILTIN_NAMES = (
    "len sum max min int float str bool abs round pow range list dict set tuple "
    "sorted reversed enumerate zip any all map filter isinstance hasattr getattr"
).split()


def normalize_feature_code(code: str) -> str:
    """Clean generated code and infer its result without rewriting intermediates."""
    lines = [line for line in code.strip().splitlines() if not line.strip().startswith(("#", "```"))]
    code = "\n".join(lines).strip()
    if not code:
        raise ValueError("Generated feature code is empty")
    tree = ast.parse(code)
    if any(
        isinstance(node, ast.Name) and node.id == "result" and isinstance(node.ctx, ast.Store)
        for node in ast.walk(tree)
    ):
        return code
    last = tree.body[-1]
    if isinstance(last, ast.Expr):
        tree.body[-1] = ast.Assign(targets=[ast.Name(id="result", ctx=ast.Store())], value=last.value)
        return ast.unparse(ast.fix_missing_locations(tree))
    if isinstance(last, ast.Assign) and len(last.targets) == 1 and isinstance(last.targets[0], ast.Name):
        if len(tree.body) == 1:
            last.targets[0].id = "result"
            return ast.unparse(tree)
        return f"{code}\nresult = {last.targets[0].id}"
    raise ValueError("Generated code must assign a feature to 'result'")


def execute_feature_code(code: str, df: pd.DataFrame, *, allow_all_missing: bool = False) -> pd.Series:
    """Execute code and require a single, row-aligned feature result."""
    if not code.strip():
        raise ValueError("Feature code is empty")
    namespace: dict[str, Any] = {
        "__builtins__": {name: getattr(builtins, name) for name in _BUILTIN_NAMES},
        "df": df.copy(deep=True),
        "np": np,
        "pd": pd,
    }
    # One namespace also exposes imports and intermediate values to lambdas/functions.
    exec(compile(code, "<feature>", "exec"), namespace, namespace)
    if "result" not in namespace:
        raise ValueError("Code does not assign to 'result' variable")
    values = namespace["result"]
    if values is None:
        raise ValueError("Feature result is None")
    if isinstance(values, pd.Series):
        if not values.index.equals(df.index):
            if (
                not values.index.is_unique
                or not df.index.is_unique
                or len(values) != len(df)
                or not df.index.isin(values.index).all()
            ):
                raise ValueError("Feature result index does not match input rows")
            values = values.reindex(df.index)
        result = values.copy()
    elif np.isscalar(values):
        result = pd.Series(values, index=df.index)
    else:
        array = np.asarray(values)
        if array.ndim != 1 or len(array) != len(df):
            raise ValueError("Feature result must be one-dimensional with one value per input row")
        result = pd.Series(array, index=df.index)
    result = result.replace([np.inf, -np.inf], np.nan)
    if not allow_all_missing and len(result) and result.isna().all():
        raise ValueError("Feature result contains only missing or non-finite values")
    return result


def validate_feature_code(code: str, sample_data: dict[str, list] | None = None) -> dict[str, Any]:
    """Return explicit syntax/output/runtime validation results."""
    result: dict[str, Any] = {"valid": True, "error": None, "warnings": []}
    try:
        tree = ast.parse(code)
        compile(tree, "<feature>", "exec")
        if not any(
            isinstance(node, ast.Name) and node.id == "result" and isinstance(node.ctx, ast.Store)
            for node in ast.walk(tree)
        ):
            raise ValueError("Code does not assign to 'result' variable")
        if sample_data is not None:
            execute_feature_code(code, pd.DataFrame(sample_data))
    except SyntaxError as error:
        result.update(valid=False, error=f"Syntax error: {error}")
    except Exception as error:
        # Generated code can raise arbitrary exception types; expose them in the result.
        result.update(valid=False, error=f"Runtime error: {error}")
    return result
