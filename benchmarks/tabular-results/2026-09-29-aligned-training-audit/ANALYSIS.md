# Aligned training-only retrospective audit

**Retrospective development, not confirmation.** Each cell receives only its corresponding original
outer-training partition. Its audit holdout and nested tuning split exactly match the prior study.
The six audit partitions overlap; they are not six independent datasets.

Methods use current benchmark run_isolated/run_case without runtime modifications.
Aligned selects among the fixed 0.1/1/10 alpha-specific feature pipelines using nested validation,
then refits the chosen feature-selection/learner parameters. Paired keeps selection alpha=1.
No choice is made from audit-holdout outcomes.

| Family | Original fold | Raw RMSE | Paired RMSE | Aligned RMSE | Paired/Aligned alpha | Paired/Aligned added |
|---|---:|---:|---:|---:|---|---|
| computer-hardware-performance | 0 | 31.182808 | 31.182808 | 31.182808 | 10/10 | 0/0 |
| computer-hardware-performance | 1 | 57.940178 | 57.940178 | 36.089309 | 1/10 | 0/5 |
| computer-hardware-performance | 2 | 39.032058 | 26.149007 | 29.520326 | 10/10 | 10/25 |
| infrared-oral-temperature | 0 | 0.25171067 | 0.25142255 | 0.23813339 | 10/10 | 1/1 |
| infrared-oral-temperature | 1 | 0.26159794 | 0.26159794 | 0.26159794 | 1/1 | 0/0 |
| infrared-oral-temperature | 2 | 1.5419319 | 1.7287452 | 2.1394429 | 10/10 | 5/50 |

## Findings

All **18/18 cells succeeded**. Aligned improves one hardware cell, worsens
another, and leaves the third unchanged relative to paired. In IR-temperature
it improves fold 0, leaves fold 1 unchanged, and further worsens fold 2.

| Family | Mean raw RMSE | Mean paired RMSE | Mean aligned RMSE | Aligned reduction vs raw | Aligned reduction vs paired |
|---|---:|---:|---:|---:|---:|
| Hardware | 42.718348 | 38.423998 | 32.264148 | +24.47% | +16.03% |
| IR-temperature | 0.685080 | 0.747255 | 0.879725 | -28.41% | -17.73% |

The IR fold-2 worsening occurs with the same chosen alpha=10 but a changed
representation: five paired additions versus fifty aligned additions.
The aligned inner losses at alpha 0.1/1/10 were
0.252882/0.241358/0.237383, selecting alpha=10 strictly from nested
training validation. The aligned audit RMSE is 2.139443 versus paired
1.728745 and raw 1.541932. This does not justify claiming that parameter
alignment alone solves extrapolation or feature-selection instability.

All six raw and six paired control cells reproduce the previous audit's
losses **exactly**, with identical chosen parameters; paired selected names
also match exactly. All eighteen aligned inner trials and all six winning
trial/selection-parameter transfers passed validation. The previous study's
32 existing artifact files remain byte-for-byte unchanged.

This comparison changes the selected pipeline, not merely alpha on a fixed
feature plan. It is descriptive retrospective evidence from overlapping
training partitions, not six independent datasets or new confirmation.
No audit outcome was used to change a search setting, feature rule or threshold.
No default promotion is justified.

## Safety and resources

- Each isolated worker payload contained only its corresponding original
  outer-training partition; the original outer-test rows were excluded.
- All six original outer split hashes, audit split hashes, nested tuning split
  hashes, outer-train position hashes and isolated-data hashes matched the
  previous study.
- Five synthetic payload-isolation/alignment tests passed.
- Fixed configuration: linear learner; alpha grid 0.1/1/10; internal seed 2027;
  audit seed 2028; three stability repeats; 1,000 candidates; 50 additions.
- Eight threads, 4,096 MiB process-tree RSS cap, 600 seconds/cell, one-hour
  global ceiling. No timeouts or failures; peak observed RSS **213.383 MiB**.
- **Complete external process elapsed: 602.741702 seconds (0.167428 hours)**,
  including startup, preparation, all isolated calls, validation and initial
  summary/artifact writing. See `launcher_timing.json`.
- Supervisor elapsed: 590.146854 seconds. Summed benchmark worker wall time:
  470.794099 seconds. Summed measured pipelines: 71.734150 seconds. These
  are nested intervals and must not be added to the external process total.
- Synthetic safety-test process: 24.675781 seconds, separate from audit elapsed.

No retained worker input pickle remains, and results contain no raw rows or
prediction vectors. No new data, live LLM, production-code or algorithm changes
were introduced by the audit.

## Integrity

Start/end/current runtime source:
`b7f0b38ae2f73780da84bb1452dcb28e1cec3bf1ec4c44487008fd2d0be2d4c1`.

Frozen script SHA256:
`fa0b92cde7c5d6d9e4ff53baf652c84f8f1d5b0f446e97b8f08e10cc071bf1b8`.

Frozen source ZIP SHA256:
`0a03740b9a4b932ba9b7c65a14ef6b72eacdf34515a90bc24d7c53a9b7318036`.

Protocol, per-cell records, aligned trial reports, source/data/split hashes,
prior-audit artifact hashes, timing records and lossless bounded archives are
preserved in this directory. `artifact_hashes.json` indexes final file checksums.
