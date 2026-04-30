# Time-Aware Tabular Prototype

A practical prototype for **leakage-safe auto feature engineering** on time-aware tabular data.

## Why this example matters

Most real feature engineering failures are not caused by weak transformations. They come from:

- random train/test splits on temporal data
- future information leaking into features
- offline features that cannot be reproduced later

This example shows a safer baseline:

1. sort by time
2. split by time
3. fit features on the training slice only
4. transform the holdout slice separately
5. compare against a plain model baseline

## Script

See:

```text
examples/time_aware_tabular_prototype.py
```

## Core pattern

```python
engineer = AutoFeatureEngineer(
    engines=["tabular"],
    max_features=30,
    selection_methods=["mutual_info", "importance"],
    correlation_threshold=0.9,
    leakage_guard="warn",
)

X_train_fe = engineer.fit_transform(
    X_train,
    y_train,
    target_name="churned",
    apply_selection=True,
)
X_test_fe = engineer.transform(X_test)
```

## Leakage guard

`AutoFeatureEngineer` now supports a lightweight `leakage_guard` option:

- `"warn"` — default, warns if suspicious columns are present
- `"raise"` — fail fast when likely leakage columns are detected
- `"off"` — disable the check

This is intentionally conservative. It does **not** prove your pipeline is safe. It just catches obvious foot-guns such as columns named like:

- `target`
- `label`
- `outcome`
- `future_*`

## Recommendation

For a real project, start with this workflow before trying more advanced LLM or agent-based feature generation. If the time-aware baseline is not trustworthy, more automation only makes the mistake faster.
