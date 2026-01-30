# FeatCopilot LLM-Powered Benchmark Report
## Overview
This report shows results using the **SemanticEngine** with GitHub Copilot integration.
When Copilot is unavailable, intelligent mock responses are used based on column semantics.

**Summary:**
- Average Improvement: **+2.15%**
- Max Improvement: **+19.66%**
- Wins: **8/12** (67%)

## Detailed Results
| Dataset | Model | Baseline | LLM FeatCopilot | Improvement | FE Time |
|---------|-------|----------|-----------------|-------------|--------|
| Medical Diagnosis (synthetic) (LLM) | LogisticRegression | 0.8533 | 0.8567 | +0.39% | 42.22s |
| Medical Diagnosis (synthetic) (LLM) | RandomForest | 0.8600 | 0.8567 | -0.39% | 42.22s |
| Medical Diagnosis (synthetic) (LLM) | GradientBoosting | 0.8600 | 0.8500 | -1.16% | 42.22s |
| Credit Risk (synthetic) (LLM) | LogisticRegression | 0.7025 | 0.7225 | +2.85% | 33.02s |
| Credit Risk (synthetic) (LLM) | RandomForest | 0.7050 | 0.7150 | +1.42% | 33.02s |
| Credit Risk (synthetic) (LLM) | GradientBoosting | 0.6975 | 0.7175 | +2.87% | 33.02s |
| Retail Demand (time series) (LLM) | Ridge | 0.7145 | 0.8550 | +19.66% | 41.81s |
| Retail Demand (time series) (LLM) | RandomForest | 0.8734 | 0.8966 | +2.65% | 41.81s |
| Retail Demand (time series) (LLM) | GradientBoosting | 0.9166 | 0.9262 | +1.05% | 41.81s |
| Product Reviews (text) (LLM) | LogisticRegression | 0.9175 | 0.8850 | -3.54% | 42.67s |
| Product Reviews (text) (LLM) | RandomForest | 0.9275 | 0.9225 | -0.54% | 42.67s |
| Product Reviews (text) (LLM) | GradientBoosting | 0.9150 | 0.9200 | +0.55% | 42.67s |

## Feature Engineering Details
| Dataset | Original Features | Engineered Features | Generation Time |
|---------|-------------------|---------------------|----------------|
| Medical Diagnosis (synthetic) (LLM) | 12 | 50 | 42.22s |
| Credit Risk (synthetic) (LLM) | 10 | 50 | 33.02s |
| Retail Demand (time series) (LLM) | 10 | 40 | 41.81s |
| Product Reviews (text) (LLM) | 6 | 42 | 42.67s |

## Notes
- LLM engine adds latency due to API calls (or mock generation)
- Column descriptions help LLM generate more relevant features
- Task descriptions provide context for domain-specific features
