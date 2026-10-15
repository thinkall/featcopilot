# Full-operator reference development comparison

**FeatCopilot is competitive with the tested linear-model references, but it
still trails full-operator inductive OpenFE on boosted-model mean gains.
This is development evidence, not a SOTA or confirmation result.**

## Protocol and coverage

The predeclared 16-family panel used three folds, seed 42, a 5,000-row cap,
linear/LightGBM learners and five representations. Fifteen families are
observational; Energy Efficiency simulation remains separate.

All **480 requested cells** were accounted for in **11.21 active hours**.
Raw, standard predictive, paired-gain predictive and full-operator inductive
OpenFE each completed 96/96 cells. Native-default AutoFeat completed 66/96:
12 timeouts, 12 upstream fitting errors reporting NaNs to LogisticRegressionCV,
and six sampled-RSS limit failures remain unscored.

The comparison used eight threads, 600 seconds/cell and a 4 GiB worker-tree RSS
cap. No source changes occurred during measurement. Paired selection was opt-in;
legacy/default selection was not changed.

## Observational primary-loss reductions versus matched raw learners

Positive is better. Classification uses log loss, regression uses RMSE.
Each cell below includes all nine classification or six regression families.

| Task / learner | Standard predictive | Paired-gain predictive | OpenFE-inductive |
|---|---|---|---|
| Classification / linear | +14.09% | +14.18% | +11.24% |
| Classification / LightGBM | -0.33% | +1.56% | +9.71% |
| Regression / linear | +10.65% | +11.37% | +8.28% |
| Regression / LightGBM | +1.62% | +0.97% | +2.56% |

Paired selection is not uniformly better: it admits useful classification
features in some boosted cases but worsens several regression outcomes relative
to the conservative rule. It remains experimental.

Native-default AutoFeat has only four complete classification families and
cannot be ranked against the nine-family classification aggregates. Its six
observational regression families are complete: +6.12% for linear models and
+1.03% for LightGBM. Output width is native, not the other methods' matched
50-addition budget.

## Largest gaps and what they actually mean

OpenFE's LightGBM classification mean is dominated by Banknote and Chronic
Kidney Disease:

| Dataset / LightGBM | Raw log loss | Paired FeatCopilot | OpenFE-inductive |
|---|---|---|---|
| Banknote | 0.015250 | 0.013493 | 0.004879 |
| Chronic Kidney Disease | 0.035504 | 0.036179 | 0.025707 |

Relative improvements can look large when raw loss is already small; the
absolute losses above provide context without changing the preregistered
metric. OpenFE's classification W/T/L is 3/1/5, versus paired FeatCopilot's
3/5/1, using the declared 1% relative-loss tie tolerance.

Inspected training-derived reference definitions reveal concrete candidate
coverage gaps: useful sums/minima/maxima in Banknote and a categorical
combination in Chronic Kidney Disease are absent from FeatCopilot's current
numeric-only candidate vocabulary. Wide-data lexicographic pair truncation
also warrants coverage improvements. These are development hypotheses for the
next opt-in implementation, not proof of their future external benefit.

Energy Efficiency separately favors OpenFE-inductive: linear loss reduction
+43.87% versus paired +39.75%, and LightGBM +8.05% versus paired -2.85%.

## Reference and runtime qualifications

OpenFE-inductive searches all supported numeric/categorical operator families,
retains its top-ranked 50 additions and freezes training statistics for
inference. It is an adapted inductive reference, not an unmodified
transductive paper reproduction.

AutoFeat-default uses native generation/selection defaults and native returned
width, with train-fitted categorical preprocessing and the disclosed
`max_gb=1` resource hint. The NaN failures were traced to **upstream AutoFeat
2.1.1 subsampling after one-hot encoding**: rare levels disappear, leaving
constant indicator columns, and unguarded zero-standard-deviation scaling
creates NaNs. The adapter supplied finite, nonconstant training input and
matched native training encoding. An unmodified native-only fixture reproduced
the failure.

Adult inner fits lost 10/8/9 indicator levels across the three folds. Online
Shoppers lost 1/4 levels in inner fits and one in the third outer refit.
Dropping indicators or imputing these internal NaNs would change the
reference, so no such correction was applied. Timeout and memory failures
also remain part of the measured resource tier. A later explicitly named
`autofeat-native` reference can omit the nondefault generation-size hint while
retaining the external worker limits; it must not replace these results.

Successful-cell mean FE time was approximately 44.6s for paired linear
selection, 23.9s for paired LightGBM, and 94.5-94.7s for OpenFE-inductive.
AutoFeat's successful-cell average excludes its failures and is not a fair
all-cohort runtime comparison by itself.

## Preserved evidence

All cell IDs were unique; failed cells contain no invented loss. Result,
manifest and source archives were verified exactly.

- [REPORT.md](REPORT.md): detailed paired results and per-family losses.
- [archive.json](archive.json): authoritative lossless result parts.
- [auxiliary_archive.json](auxiliary_archive.json): exact source ZIP and
  manifest parts, kept below the repository's per-file size limit.

The uncompressed `results.json`, `manifest.json` and whole `source.zip` remain
local for exact resume; the public numbered archives preserve their content.
To reconstruct an auxiliary artifact:

```python
import gzip
import json
from pathlib import Path

root = Path(".")
index = json.loads((root / "auxiliary_archive.json").read_text())
for artifact in index["files"]:
    packed = b"".join((root / part).read_bytes() for part in artifact["parts"])
    content = gzip.decompress(packed) if artifact["compression"] == "gzip" else packed
    # content is the exact original manifest JSON or source ZIP byte stream.
```
