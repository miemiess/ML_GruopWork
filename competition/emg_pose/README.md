# EMGInsight · Neural Interfaces 2026 Track 4

**Competition team** — Ma Mingyang, Han Yuchen, Zhou Ying

[Codabench competition](https://www.codabench.org/competitions/17984/#/participate-tab) · [Official submission guide](https://www.codabench.org/competitions/17984/participate.md)

This entry uses the EMGInsight RMS, MAV and waveform length features with standardized multi-output ridge regression. It maps independent 5-second windows of 16-channel wrist EMG at 2 kHz to 20 UmeTrack joint-angle trajectories in radians. Each output step represents one 50 ms bin, which the evaluator expands to the target length. The model uses no state across windows. Both training routes below use public EMG2Pose only, with no additional corpus or pretrained model. The classroom gesture, force and fatigue datasets do not supply the joint-angle targets needed here.

## Reproduce the packaged mini model

The packaged entry was trained on Meta's [public EMG2Pose mini archive](https://github.com/facebookresearch/emg2pose#getting-started-small-sanity-check-dataset). The archive contains both hands across 15 recordings from a single participant, all marked as validation in the full corpus. For this entry, recordings 2 and 3 were excluded from training. Recording 3 supplied the local check, with **26.69 degrees MAE on 10 valid windows**. This check measures same-participant performance.

Run from the root of the official competition kit at commit `8447e18119d5116de5d3f0232d467e326774f88a` with Benchopt, PyTorch, NumPy and h5py installed. Set `PROJECT_DIR` to the absolute path of this repository.

```bash
curl -fLO https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_dataset_mini.tar
tar xf emg2pose_dataset_mini.tar
PYTHONPATH="$PWD" OPENBLAS_NUM_THREADS=1 python \
  "$PROJECT_DIR/competition/emg_pose/train_mini.py" \
  emg2pose_dataset_mini "$PROJECT_DIR/EMGInsight-Track4.zip"
```

Archive SHA-256 — `d7400e98508ccbb2139c2d78e552867b23501f637456546fd6680f3fe7fec50d`. The model fits a single pass of ridge sufficient statistics. A CPU is sufficient and the training and local evaluation take seconds after extraction.

## Optional full-corpus training and export

Run these commands in Bash. Set `PROJECT_DIR` to the absolute path of this repository. Use a fresh Python environment with the official kit's dependencies.

```bash
git clone https://github.com/neural-interfaces26/2026-competition.git
cd 2026-competition
git checkout 8447e18119d5116de5d3f0232d467e326774f88a
python -m pip install -r requirements.txt

PROJECT_DIR=/absolute/path/to/ML_GruopWork
SOLVER="$PROJECT_DIR/competition/emg_pose/submission.py"
EXPORT_DIR="$PWD/tracks/emg_pose/outputs/EMGInsight-RMS-MAV-WL-Ridge"

benchopt prepare tracks/emg_pose -d Salter2024Emg2pose
benchopt run tracks/emg_pose -d Salter2024Emg2pose -s "$SOLVER" \
  -o 'EMG-pose[training=True]' --seed 42
```

The training run exports `submission.py` and `weights.npz` and reports public-split evaluation. Verify that the exported model can be reloaded and evaluated before using a sealed submission slot.

```bash
COMPET_SUBMISSION_DIR="$EXPORT_DIR" benchopt run tracks/emg_pose \
  -d Salter2024Emg2pose -s "$EXPORT_DIR/submission.py"

zip -j "$PROJECT_DIR/EMGInsight-Track4.zip" \
  "$EXPORT_DIR/submission.py" "$EXPORT_DIR/weights.npz"
unzip -l "$PROJECT_DIR/EMGInsight-Track4.zip"
```

The two files must appear at the ZIP root. Under **My Submissions**, select **Sealed Final Phase** and upload `EMGInsight-Track4.zip` from an approved account associated with your team. Check the status for **Finished**. Codabench organization membership controls team attribution and quotas independently of the roster documented here.
