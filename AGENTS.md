# Copilot Agent Configuration

## Permissions

- Allow all file edits without confirmation
- Allow GitHub commit tools (create commits, push, etc.)
- Allow running shell commands

## Pre-commit Hook

**IMPORTANT:** Always run `pre-commit run --all-files` before creating any git commit. Do not skip this step.

## Workflow Guidelines

1. When making code changes, edit files directly without asking for permission
2. Before committing changes:
   - Run `pre-commit run --all-files` and fix any issues
   - Only proceed with commit after pre-commit passes
3. Use descriptive commit messages following conventional commits format
4. **Git add only the files you changed** - Do NOT use `git add -A` or `git add .`
   - Multiple sessions may be working on this project simultaneously
   - Use explicit file paths: `git add path/to/file1.py path/to/file2.py`
   - This prevents accidentally committing changes from other sessions

## Safety Guardrails

**NEVER do the following without explicit user approval:**

- `git push --force` or `git push -f` - Always use regular push; if rejected, ask user for guidance
- Remove/delete folders or directories - Always ask for permission first
- Delete multiple files at once - Confirm with user before bulk deletions
- Run destructive commands (e.g., `rm -rf`, `git reset --hard`)

## Coding Standards

### Code Style

This project uses **Black** and **Ruff** for code formatting and linting.

- **Line length:** 120 characters max
- **Python version:** 3.9+ (use modern type hints)
- **Quotes:** Double quotes for strings
- **Imports:** Organized in order: stdlib, third-party, local (enforced by ruff)
- **Type hints:** Use Python 3.9+ style (`list[str]` not `List[str]`, `dict[str, Any]` not `Dict[str, Any]`)
- **Optional types:** Use `Optional[X]` or `X | None` for nullable types

#### Import Organization

```python
"""Module docstring."""

import asyncio                              # 1. Standard library
from typing import Any, Optional, Union

import numpy as np                          # 2. Third-party
import pandas as pd
from pydantic import BaseModel, Field

from featcopilot.core.base import BaseEngine  # 3. Local imports
from featcopilot.utils.logger import get_logger

logger = get_logger(__name__)               # 4. Module-level logger
```

#### Class Structure

```python
class MyClass(BaseClass):
    """
    Brief class description.

    Longer description if needed.

    Parameters
    ----------
    param1 : type
        Description
    param2 : type, optional
        Description

    Examples
    --------
    >>> obj = MyClass(param1='value')
    >>> result = obj.method()
    """

    def __init__(self, param1: str, param2: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.param1 = param1
        self.param2 = param2
        self._private_attr = None  # Private attributes prefixed with _
```

### Docstring Style

Use **NumPy-style docstrings** for all public classes, methods, and functions.

#### Function/Method Docstring

```python
def function_name(
    param1: str,
    param2: int,
    optional_param: Optional[float] = None,
) -> ReturnType:
    """
    Brief one-line description.

    Longer description if needed, explaining what the function does,
    any important details, or caveats.

    Parameters
    ----------
    param1 : str
        Description of param1
    param2 : int
        Description of param2
    optional_param : float, optional
        Description of optional parameter

    Returns
    -------
    ReturnType
        Description of return value

    Raises
    ------
    ValueError
        When invalid input is provided

    Examples
    --------
    >>> result = function_name('value', 42)
    >>> print(result)
    """
```

#### Short Docstrings

For simple/obvious methods, use single-line docstrings:

```python
def get_feature_names(self) -> list[str]:
    """Get names of generated features."""
    return self._feature_names.copy()
```

### Logging

- **Never use `print()` for logging** - Use the dedicated logger module instead
- Import the logger at module level after imports:
  ```python
  from featcopilot.utils.logger import get_logger

  logger = get_logger(__name__)
  ```
- Use appropriate log levels:
  - `logger.debug()` - Detailed diagnostic information
  - `logger.info()` - General operational messages
  - `logger.warning()` - Warning messages (e.g., fallbacks, deprecations)
  - `logger.error()` - Error messages
  - `logger.critical()` - Critical failures

### Pydantic Models

Use Pydantic for configuration classes:

```python
class EngineConfig(BaseModel):
    """Configuration for feature engineering engines."""

    name: str = Field(description="Engine name")
    enabled: bool = Field(default=True, description="Whether engine is enabled")
    max_features: Optional[int] = Field(default=None, description="Max features to generate")
    verbose: bool = Field(default=False, description="Verbose output")
```

### Error Handling

- Use specific exception types
- Include helpful error messages with context:
  ```python
  if not self._is_fitted:
      raise RuntimeError("Selector must be fitted before transform")

  if method not in self.METHODS:
      raise ValueError(f"Method must be one of {self.METHODS}")
  ```

### Naming Conventions

- **Classes:** PascalCase (`ImportanceSelector`, `TabularEngine`)
- **Functions/methods:** snake_case (`fit_transform`, `get_feature_names`)
- **Variables:** snake_case (`feature_scores`, `is_fitted`)
- **Constants:** UPPER_SNAKE_CASE (`DEFAULT_MODEL`, `SUPPORTED_METHODS`)
- **Private attributes/methods:** Prefix with `_` (`_is_fitted`, `_validate_input`)

## Additional Configurations

<!-- Add more configurations below as needed -->
