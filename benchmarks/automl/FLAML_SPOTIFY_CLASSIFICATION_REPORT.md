# FLAML Spotify Genre Classification Benchmark Report
## Overview
This benchmark evaluates FLAML AutoML performance on the Spotify tracks dataset
for multi-class genre classification (4 genres).

- **Dataset**: `maharshipandya/spotify-tracks-dataset`
- **Genres**: pop, acoustic, hip-hop, punk-rock
- **Time budget**: 180s per FLAML run
- **FeatCopilot time**: 0.2s

### Features Used
- **Baseline**: Numeric features only (15 features)
- **FeatCopilot**: Numeric + text features + target-encoded artists

## Summary
| Metric | Baseline | +FeatCopilot | Improvement |
|--------|----------|--------------|-------------|
| Accuracy | 0.8150 | 0.8375 | +2.76% |
| F1 (macro) | 0.8158 | 0.8388 | - |
| F1 (weighted) | 0.8158 | 0.8388 | +2.82% |
| Train Time | 181.6s | 183.0s | - |
| Features | 15 | 32 | +17 |

## Model Details
- **Baseline best model**: xgb_limitdepth
- **FeatCopilot best model**: xgb_limitdepth

## Key Findings
- FeatCopilot improved accuracy by **+2.76%**
- Feature engineering added 17 features (15 → 32)
- Total FeatCopilot overhead: 0.2s

**✅ SIGNIFICANT IMPROVEMENT: FeatCopilot added +2.76% accuracy**
