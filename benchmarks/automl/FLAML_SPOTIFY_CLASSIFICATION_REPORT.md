# FLAML Spotify Genre Classification Benchmark Report
## Overview
This benchmark evaluates FLAML AutoML performance on the Spotify tracks dataset
for multi-class genre classification (4 genres).

- **Dataset**: `maharshipandya/spotify-tracks-dataset`
- **Genres**: pop, acoustic, hip-hop, punk-rock
- **Time budget**: 480s per FLAML run
- **FeatCopilot time**: 41.2s

### Features Used
- **Baseline**: All original features (18 features)
- **FeatCopilot**: All features + LLM-generated + text features + target encoding

## Summary
| Metric | Baseline | +FeatCopilot | Improvement |
|--------|----------|--------------|-------------|
| Accuracy | 0.9287 | 0.9337 | +0.54% |
| F1 (macro) | 0.9289 | 0.9339 | - |
| F1 (weighted) | 0.9289 | 0.9339 | +0.54% |
| Train Time | 507.5s | 600.6s | - |
| Features | 18 | 40 | +22 |

## Model Details
- **Baseline best model**: catboost
- **FeatCopilot best model**: catboost

## Key Findings
- FeatCopilot improved accuracy by **+0.54%**
- Feature engineering added 22 features (18 → 40)
- Total FeatCopilot overhead: 41.2s

**✅ TARGET ACHIEVED: F1-score 0.9339 >= 0.85**
