# Auto Feature Engineering Use-Case Benchmark

Compares a plain baseline with FeatCopilot and common automatic feature engineering tools on an interaction-heavy tabular classification task.

| Tool | Status | ROC-AUC | Feature Count |
|------|--------|---------|---------------|
| baseline | ok | 0.6330 | 9 |
| featcopilot | ok | 0.6328 | 11 |
| featuretools | ok | 0.6362 | 60 |
| autofeat | failed: check_array() got an unexpected keyword argument 'force_all_finite' | - | - |
