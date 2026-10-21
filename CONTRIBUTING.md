# Contributing to EMGInsight

All six members share ownership of the project. Choose work through Issues; roles are not permanently assigned.

## Start a contribution

Create an Issue describing the problem, proposed output, affected modules, data assumptions, and acceptance criteria. Agree on interface changes before parallel implementation.

```bash
git switch main
git pull --ff-only origin main
git switch -c feature/short-description
```

Use `fix/` or `docs/` when appropriate. Commit focused changes and push your branch:

```bash
git add path/to/changed-file
git commit -m "Describe the concrete change"
git push -u origin feature/short-description
```

Open a PR targeting `main`. Include the linked Issue, purpose, checks performed, results, and limitations. Another member should review it before merge. Resolve conflicts on the feature branch and rerun affected checks. Do not force-push shared `main`.

## Implementation conventions

- Put reusable logic in `src/emginsight/`; notebooks should call package functions.
- Use separate adapters for gesture, force, and fatigue data.
- Preserve subject, session, recording, channel, timestamp, and label provenance.
- Adopt `X: (n_windows, n_features)` and `y: (n_windows,)` or `(n_windows, n_outputs)`, plus group IDs. Describe shapes and units in docstrings.
- Models should expose `fit(X, y)` and `predict(X)`; document classification labels and regression output shapes.
- Write core ML algorithms in Python/NumPy. Clearly identify library reference models; use no deep learning.
- Follow the [research protocol](docs/research_protocol.md) for leakage prevention and evaluation.

## Before requesting review

- Run relevant tests once implemented; include the exact command and outcome in the PR.
- For algorithm changes, add meaningful synthetic/known-answer, edge-case, and reference-comparison tests.
- For data changes, check dimensions, labels, timestamp alignment, group disjointness, and training-only fitted transforms.
- Restart and run changed notebooks in order. Remove private paths, raw-record dumps, and large outputs.
- Update setup/configuration documentation when behavior changes.
- Inspect the diff for secrets, raw data, participant metadata, model binaries, or unsupported claims.
- Keep small reviewed metrics/figures under `results/summaries/` and `results/figures/`; put large local artifacts in `results/runs/`.

No automated CI or branch protection is configured by this scaffold. Reviewers should check the stated validation evidence. Do not report unimplemented checks as passing.
