# Research protocol

This is a starting agreement. Dataset-specific choices require an audit and a recorded team decision.

## Shared interfaces

Adapters should return signals, timestamps/sampling information, targets, channel names/units, and subject/session/recording IDs. Window outputs should preserve source recording and start/end time.

Feature datasets use `X: (n_windows, n_features)`, classification `y: (n_windows,)`, or regression `y: (n_windows, n_outputs)`, plus subject/session/recording groups. Models expose `fit(X, y)` and `predict(X)`. Separate training, prediction, and evaluation code.

## Prevent leakage

1. Inventory source recordings and group IDs first.
2. For unseen-user evaluation, assign whole subjects to train/validation/test. A session-only holdout is a separate, weaker protocol and must be named accordingly.
3. Split before windowing. Never let overlapping windows or the same recording cross partitions.
4. Fit normalization, imputation, feature/channel selection, and any learned preprocessing only on each training fold. Process recordings independently so filtering does not cross split boundaries.
5. Choose hyperparameters and channel subsets using grouped validation only. Keep a frozen test partition, or use nested grouped evaluation when appropriate.
6. Assert group disjointness and save the exact split IDs.

If a class is absent from a fold, report support and handling explicitly rather than silently dropping failures. Small cohorts may require fewer folds.

## Features and models

Start with RMS, mean absolute value (MAV), and waveform length (WL). Record window duration, stride, sampling rate, channel ordering, filtering choices, feature formulas, and target aggregation/alignment. Choose these per dataset; no universal window is assumed.

Implement core algorithms in Python/NumPy, including training and prediction. Label scikit-learn comparisons as references. Candidate models are proposals until validated. Include majority-class and training-mean baselines.

## Evaluation

- Classification: macro-F1 as the main score; per-class precision/recall/F1, class support, and confusion matrices. Accuracy is supplementary.
- Regression: MAE and RMSE per selected target, target units, plus a stated aggregation rule. Compare predictions and measured signals visually.
- Generalization: report held-out participant identities and per-user variability where feasible.
- Channel study: compare full and reduced subsets within the same dataset using identical splits and budgets. Select subsets on training/validation only.
- Record seeds, runtime, failures, exclusions, and uncertainty; never select a model from test-set performance.

## Interpretation

Keep all three studies separate. Differences in acquisition and population prevent direct sensor-count conclusions across datasets. Discuss label subjectivity, cohort coverage, domain shift, and failure cases alongside social-good motivations.
