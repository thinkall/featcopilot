# PR Summary — feat/strengthen-core-api

## What this branch changes

This branch improves FeatCopilot in three layers:

1. **Core API reliability**
   - safer source imports when package metadata is unavailable
   - `AutoFeatureEngineer.fit_transform()` now forwards transform-time kwargs
   - earlier config validation for engines / selection methods / leakage guard

2. **Safety and practical workflow support**
   - lightweight leakage guard in `AutoFeatureEngineer`
   - time-aware prototype example for leakage-safe evaluation
   - relational engine key validation and duplicate relationship deduping
   - optional per-row mode for `TimeSeriesEngine`

3. **Benchmark realism and use-case coverage**
   - more realistic split logic in tool-comparison benchmarks
   - targeted benchmark for interaction-heavy auto feature engineering use cases

## Commits

- `8e5d323` — `fix: harden source imports and fit_transform kwargs`
- `50b399c` — `feat: add leakage guard and time-aware prototype`
- `50957fa` — `feat: improve engine APIs and benchmark realism`

## Highlights

### AutoFeatureEngineer
- new `leakage_guard` modes: `warn`, `raise`, `off`
- early validation of:
  - engine names
  - selection methods
  - positive `max_features`
- transform-time kwargs are preserved in one-shot workflows

### TimeSeriesEngine
- new `series_in_rows` mode for row-wise sequence cells
- clearer `time_column` validation
- default behavior preserved for current aggregate-style usage

### RelationalEngine
- duplicate relationships no longer accumulate
- missing child/parent keys fail fast with useful errors

### Benchmarks
- classification splits use stratification when possible
- forecast/time-series splits avoid random shuffling
- added focused use-case benchmark:
  - `benchmarks/use_cases/run_auto_feature_engineering_benchmark.py`

### Docs / examples
- time-aware tabular example
- relational feature engineering example

## Validation

- targeted tests passed
- full test suite passed:
  - `635 passed, 2 skipped`
- pre-commit passed

## Suggested PR title

**Improve core API reliability, add leakage guard, and strengthen benchmark realism**

## Suggested reviewer notes

The main intended behavior changes are:
- earlier user-facing validation
- new optional leakage warnings/errors
- new optional row-wise time-series mode
- slightly more realistic benchmark split policy

The branch is intentionally incremental rather than a redesign.
