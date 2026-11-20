"""Train and package EMGInsight using Meta's public EMG2Pose mini archive."""

import argparse
import csv
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import h5py
import numpy as np
import torch

from benchmark_utils.data import resample_labels
from submission import Solver


N_TIMES = 10_000
N_CHANNELS = 16
N_JOINTS = 20
SFREQ = 2_000


def batches(files):
    for path in files:
        with h5py.File(path, "r") as handle:
            group = handle["emg2pose"]
            if group.attrs["sample_rate"] != SFREQ:
                raise ValueError(f"Unexpected sample rate in {path}")
            rows = group["timeseries"][:]

        n = len(rows) // N_TIMES
        rows = rows[: n * N_TIMES].reshape(n, N_TIMES)
        emg, angles = rows["emg"], rows["joint_angles"]
        valid = (
            np.isfinite(emg).all(axis=(1, 2))
            & np.isfinite(angles).all(axis=(1, 2))
            & (~np.isclose(angles, 0).all(axis=2)).all(axis=1)
        )
        if valid.any():
            yield (
                torch.from_numpy(emg[valid].transpose(0, 2, 1)),
                torch.from_numpy(angles[valid].transpose(0, 2, 1)),
                None,
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Extracted emg2pose_dataset_mini")
    parser.add_argument("output", type=Path, help="Destination submission ZIP")
    args = parser.parse_args()

    files = sorted(args.dataset.glob("*.hdf5"))
    with (args.dataset / "metadata.csv").open(newline="") as handle:
        metadata = {row["filename"]: row for row in csv.DictReader(handle)}
    if len(files) != 30 or any(metadata[p.stem]["split"] != "val" for p in files):
        raise ValueError("Expected the 30 recordings in Meta's public mini archive.")

    # Reserve both hands of recordings 2 and 3 (Meta's mini validation/test).
    held_out = [p for p in files if "-recording-3_" in p.stem]
    train = [p for p in files if not any(f"-recording-{i}_" in p.stem for i in (2, 3))]
    with TemporaryDirectory() as temp:
        directory = Path(temp)
        meta = dict(
            n_chans=N_CHANNELS, n_times=N_TIMES, n_joints=N_JOINTS,
            sfreq=SFREQ, device="cpu", submission_dir=directory,
        )
        solver = Solver()
        solver.meta = meta
        solver.train_loader = batches(train)
        model = solver.load_model(meta)
        solver.fit(model, solver.train_loader)
        solver.save_model(model, directory)
        model = Solver().load_model(meta)

        error = 0.0
        count = 0
        for X, y, _ in batches(held_out):
            prediction = model.predict(X)
            if prediction.shape != (len(X), N_JOINTS, 100):
                raise ValueError(f"Unexpected prediction shape {prediction.shape}")
            if not np.isfinite(prediction).all():
                raise ValueError("Nonfinite held-out predictions")
            expanded = resample_labels(prediction, N_TIMES)
            error += np.abs(expanded - y.numpy()).sum(dtype=np.float64)
            count += y.numel()
        if not count:
            raise ValueError("The mini test recordings contained no valid windows.")
        print(f"Mini held-out MAE: {np.degrees(error / count):.2f} deg "
              f"({count // (N_JOINTS * N_TIMES)} windows; same participant)")

        with ZipFile(args.output, "w", ZIP_DEFLATED) as archive:
            archive.write(Path(__file__).with_name("submission.py"), "submission.py")
            archive.write(directory / "weights.npz", "weights.npz")


if __name__ == "__main__":
    main()
