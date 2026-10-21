# EMGInsight
### Decoding muscle signals with traditional machine learning

**61.501 Production Machine Learning · Group 12 · PML 2026**

EMGInsight studies how public surface electromyography (sEMG) recordings can support hand-gesture recognition, finger-force estimation, and muscle-fatigue classification. We are building a shared, reproducible framework to compare traditional machine learning models, investigate channel reduction, and measure generalization to unseen users.

**Status:** project scaffold. Data adapters, feature extraction, models, training pipelines, and the offline demo are planned; no experiment results or trained models are included yet. The [project proposal](Group_12_EMGInsight_Visual_Proposal_Revised.pdf) remains the project reference. All three studies are subject to data feasibility.

[Dataset guide](data/README.md) · [Research protocol](docs/research_protocol.md) · [Contributing](CONTRIBUTING.md) · [Experiment record](docs/experiment_template.md)

## Project requirements

- **No deep learning.** Use traditional classification and regression methods.
- **Implement core ML algorithms ourselves in Python/NumPy.** Library models may provide reference comparisons, but must be clearly labeled and must not replace the custom implementations.
- **Address three dataset-specific tasks.** Share reusable utilities while keeping dataset adapters, labels, and evaluation protocols separate.
- **Evaluate prediction quality, channel subsets, and unseen-user generalization.** Compare simple baselines and custom models using the same splits.
- **Make experiments reproducible.** Record dataset versions, file checksums, split identities, configurations, random seeds, environments, and commands.
- **Explain social benefit and limitations.** Deliver evidence appropriate to offline research.

Planned deliverables: reproducible code, custom ML implementations, three feasibility-dependent studies, model/channel comparisons, evaluation plots, an offline demonstration, and a final report.

## Three datasets, three tasks

| Study | Public source and input | Target | Key limits |
| --- | --- | --- | --- |
| Gesture recognition | [UCI EMG Data for Gestures](https://archive.ics.uci.edu/dataset/481/emg+data): 8 forearm channels, 36 participants | Hand-gesture class | Label 0 means unmarked data; class 7 was not performed by everyone. Specify inclusion rules and avoid treating adjacent windows as independent users. |
| Finger-force estimation | [putEMG-Force](https://biolab.put.poznan.pl/putemg-dataset/): 24 EMG channels with measured force | Selected measured force signal(s) | Use `emg_force` trials. Measured `FORCE_*` columns differ from requested `TRAJ_*` trajectories. Audit target mapping, units, calibration, and available participants/sessions before modeling. |
| Muscle-fatigue classification | [Zenodo sEMG and Self-Perceived Fatigue v2](https://zenodo.org/records/14182446): 8 muscles, 13 healthy adults | Self-perceived fatigue level 0/1/2 | EMG and ratings have different sampling rates and require timestamp alignment. Labels are subjective; the small healthy cohort limits broader claims. |

These datasets differ in devices, anatomy, protocols, and labels. Compare channel subsets **within each dataset**; channel counts across datasets do not establish sensor superiority. Keep separate models and results for each task. See the [data guide](data/README.md) for acquisition, provenance, and redistribution rules.

## Research and evaluation

```text
Public digitized sEMG + metadata
  -> user/session split
  -> preprocessing and windowing
  -> RMS / MAV / waveform length features
  -> custom traditional ML + reference baselines
  -> held-out evaluation and channel comparisons
```

Start with simple baselines (majority class for classification; training-target mean for regression). Candidate custom models include k-nearest neighbors, logistic regression, and ridge regression; final choices will be recorded per experiment.

Use macro-F1, per-class scores, and confusion matrices for classification; use MAE and RMSE for regression, with target units and per-output scores. Report per-user results and variability where feasible. Hold out users for the primary generalization study, with session-based studies clearly labeled. Split recordings before windowing; fit scalers, feature selection, and other learned transforms on training data only. Tune on grouped validation data and evaluate the test set after choices are fixed. See the [research protocol](docs/research_protocol.md).

## AI for social good

Potential applications include accessible prosthetic-hand interfaces, rehabilitation research, athlete/workplace muscle monitoring, and alternative VR interaction. Channel-reduction studies may inform simpler sensing setups. These are research motivations, not demonstrated outcomes. This project is an offline study and makes no clinical diagnostic or device-control claims. Discuss participant coverage, failure cases, and accessibility limits in the final report.

## Team and collaboration

**Members:** Ma Mingyang · Han Yuchen · Zhou Yin · Wei Xin · Gao Ruiyan · Li Ji

We collaboratively build one system in one repository. There are **no fixed role allocations**. Members choose work through Issues and can contribute across modules. Available work includes the three task studies, preprocessing/features, custom algorithms/testing, evaluation, integration, documentation, and the demo.

1. Agree on an Issue with scope, interfaces, and acceptance criteria.
2. Create a feature branch from the latest `main`, such as `feature/gesture-baseline`.
3. Make a focused change, document it, and run relevant checks.
4. Open a PR into `main`, including validation evidence and any data assumptions.
5. Ask another member to review; address feedback before merging.

The shared `main` should stay usable. Avoid routine direct pushes to it. This is a team convention; this scaffold does not configure enforced branch protection. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Workspace

```text
.
├── docs/                       # Protocol, decisions, experiment-record template
├── data/                       # Acquisition/provenance guide; local data stays ignored
│   ├── raw/                    # Immutable source files
│   ├── interim/                # Cleaned/aligned intermediate files
│   ├── processed/              # Features, labels, group/split metadata
│   └── manifests/              # Reviewed provenance and checksum records
├── src/emginsight/              # Shared Python package
│   ├── datasets/               # One adapter per dataset
│   ├── preprocessing/          # Windowing, filtering, training-only transforms
│   ├── features/               # Shared feature extraction
│   ├── models/                 # Custom traditional ML implementations
│   └── evaluation/             # Grouped splits, metrics, comparison helpers
├── notebooks/experiments/       # Exploratory notebooks; reuse package functions
├── configs/                    # Versioned experiment specifications
├── tests/                      # Future algorithm, leakage, and adapter tests
├── results/                    # Reviewed summaries/figures; large artifacts ignored
├── demo/                       # Planned offline inference demonstration
├── CONTRIBUTING.md
├── pyproject.toml              # Package metadata and tool settings
└── requirements.txt            # Starter development installation
```

The proposal PDF remains at the repository root to preserve the existing reference. Package folders contain planning notes rather than model implementations.

## Setup

Use **Python 3.11 or newer**. From a terminal:

```bash
git clone https://github.com/miemiess/ML_GruopWork.git
cd ML_GruopWork
python -m venv .venv
```

Activate the environment:

| Terminal | Command |
| --- | --- |
| Windows PowerShell | `.venv\Scripts\Activate.ps1` |
| Windows Command Prompt | `.venv\Scripts\activate.bat` |
| macOS/Linux | `source .venv/bin/activate` |

If PowerShell blocks activation, use `.venv\Scripts\python.exe` instead of `python` in the following commands.

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import emginsight; print(emginsight.__version__)"
python -m pip check
```

This installs the editable package and starter tools for data analysis, notebooks, reference models, and testing. Dataset-specific readers may need extra dependencies after file-format inspection. To explore locally, run `python -m jupyterlab` and use `notebooks/experiments/`.

There is no training or demo command yet. `python -m pytest` becomes the test command once tests are added; at scaffold stage it reports no tests collected (exit code 5). See [tests/README.md](tests/README.md).

## Reproducing an experiment

1. Obtain data from the original sources using [data/README.md](data/README.md); keep raw files local and unchanged.
2. Copy [configs/experiment.example.toml](configs/experiment.example.toml), fill dataset-specific choices, and record provenance.
3. Save exact subject/session split IDs, label rules, window settings, channel IDs, preprocessing, model settings, and seeds.
4. Record the Git commit and environment with `python --version` and `python -m pip freeze` in a local run folder.
5. When a pipeline exists, record its exact command and metrics using the [experiment template](docs/experiment_template.md). Commit only reviewed summaries and figures.

Dependency ranges in the starter configuration are not an exact environment lock. Archive exact resolved versions for each reported run; a shared lock can be adopted when the first pipeline is validated.

## Next milestones

- Audit a small sample from each source, confirm licenses and label availability, and document feasibility.
- Agree on adapters and grouped splits; implement preprocessing and feature extraction.
- Implement and test custom models, then establish task-specific baselines.
- Compare channels/models, prepare the offline demo, and write the final report.

Dataset rights remain with their original providers. No project-wide code license has been chosen yet.
