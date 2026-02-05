# FLAML Spotify Genre Classification Benchmark Report
## Overview
This benchmark evaluates FLAML AutoML performance on the Spotify tracks dataset
for multi-class genre classification (4 genres).

- **Dataset**: `maharshipandya/spotify-tracks-dataset`
- **Genres**: pop, acoustic, hip-hop, punk-rock
- **Time budget**: 480s per FLAML run
- **FeatCopilot time**: 40.3s

### Features Used
- **Baseline**: Numeric features only (15 features)
- **FeatCopilot**: Numeric + LLM-generated + text features + target encoding

## Summary
| Metric | Baseline | +FeatCopilot | Improvement |
|--------|----------|--------------|-------------|
| Accuracy | 0.8237 | 0.9263 | +12.44% |
| F1 (macro) | 0.8243 | 0.9263 | - |
| F1 (weighted) | 0.8243 | 0.9263 | +12.37% |
| Train Time | 480.4s | 520.3s | - |
| Features | 15 | 50 | +35 |

## Model Details
- **Baseline best model**: lgbm
- **FeatCopilot best model**: catboost

## Key Findings
- FeatCopilot improved accuracy by **+12.44%**
- Feature engineering added 35 features (15 → 50)
- Total FeatCopilot overhead: 40.3s

**✅ SIGNIFICANT IMPROVEMENT: FeatCopilot added +12.44% accuracy**
