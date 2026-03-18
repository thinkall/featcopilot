# Auto Feature Engineering Use-Case Benchmark

Compares a plain baseline with FeatCopilot and common automatic feature engineering tools on an interaction-heavy tabular classification task.

| Tool | Status | ROC-AUC | Feature Count |
|------|--------|---------|---------------|
| baseline | ok | 0.7440 | 9 |
| featcopilot | ok | 0.7563 | 15 |
| featuretools | failed: 'NoneType' object has no attribute 'columns' | - | - |
| autofeat | failed: 'Series' object has no attribute 'ravel' | - | - |
