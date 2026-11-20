"""EMGInsight RMS/MAV/WL ridge model for Codabench Track 4."""

import os
from pathlib import Path

import numpy as np

from benchmark_utils.base_solver import CompetSolver
from benchmark_utils.data import to_numpy


class PoseRidge:
    def __init__(self, n_times, sfreq):
        # Equal-width bins align the coarser output with nearest resampling.
        bins = max(1, round(n_times / (sfreq * 0.05)))
        while n_times % bins:
            bins -= 1
        self.bins = bins
        self.weights = None
        self.intercept = None

    def features(self, X):
        X = np.asarray(to_numpy(X), dtype=np.float32)
        batch, channels, _ = X.shape
        chunks = X.reshape(batch, channels, self.bins, -1)
        rms = np.sqrt(np.mean(chunks ** 2, axis=-1))
        mav = np.mean(np.abs(chunks), axis=-1)
        wl = np.sum(np.abs(np.diff(chunks, axis=-1)), axis=-1)
        return np.stack((rms, mav, wl), axis=-1).transpose(0, 2, 1, 3).reshape(
            batch, self.bins, channels * 3
        )

    def predict(self, X):
        if self.weights is None:
            raise RuntimeError("Train the Track 4 model before inference.")
        angles = self.features(X) @ self.weights + self.intercept
        return angles.transpose(0, 2, 1).astype(np.float32)


class Solver(CompetSolver):
    name = "EMGInsight-RMS-MAV-WL-Ridge"

    def load_model(self, meta):
        model = PoseRidge(meta["n_times"], meta["sfreq"])
        path = Path(meta["submission_dir"]) / "weights.npz"
        if path.is_file():
            with np.load(path, allow_pickle=False) as checkpoint:
                model.weights = checkpoint["weights"]
                model.intercept = checkpoint["intercept"]
            if (
                model.weights.shape != (meta["n_chans"] * 3, meta["n_joints"])
                or model.intercept.shape != (meta["n_joints"],)
                or not np.isfinite(model.weights).all()
                or not np.isfinite(model.intercept).all()
            ):
                raise ValueError("Checkpoint does not match Track 4 metadata.")
        elif getattr(self, "train_loader", None) is None:
            raise FileNotFoundError(path)
        return model

    def fit(self, model, train_loader):
        n_features = self.meta["n_chans"] * 3
        n_joints = self.meta["n_joints"]
        # Accumulate sufficient statistics without retaining entire recordings.
        xtx = np.zeros((n_features + 1, n_features + 1))
        xty = np.zeros((n_features + 1, n_joints))

        for X, y, _info in train_loader:
            features = model.features(X).reshape(-1, n_features)
            targets = np.asarray(to_numpy(y), dtype=np.float64)
            if targets.shape != (len(X), n_joints, X.shape[-1]):
                raise ValueError("Expected one radian angle per joint and EMG sample.")
            targets = targets.reshape(len(X), n_joints, model.bins, -1).mean(axis=-1)
            targets = targets.transpose(0, 2, 1).reshape(-1, n_joints)
            if not np.isfinite(features).all() or not np.isfinite(targets).all():
                raise ValueError("Nonfinite values in training samples.")
            design = np.ones((len(features), n_features + 1))
            design[:, :-1] = features
            xtx += design.T @ design
            xty += design.T @ targets

        count = xtx[-1, -1]
        if count == 0:
            raise ValueError("The training split contained no windows.")

        # Center and standardize before ridge, with an unpenalized intercept.
        feature_sum = xtx[:-1, -1]
        mean = feature_sum / count
        target_mean = xty[-1] / count
        covariance = xtx[:-1, :-1] - np.outer(feature_sum, mean)
        scale = np.sqrt(np.maximum(covariance.diagonal() / count, 0))
        scale[scale < 1e-12] = 1
        gram = covariance / np.outer(scale, scale)
        gram.flat[:: n_features + 1] += 10.0
        cross = (xty[:-1] - np.outer(feature_sum, target_mean)) / scale[:, None]
        model.weights = np.linalg.solve(gram, cross) / scale[:, None]
        model.intercept = target_mean - mean @ model.weights

    def save_model(self, model, path):
        # Store numeric arrays only, without pickled Python dependencies.
        tmp = path / "weights.tmp.npz"
        np.savez_compressed(tmp, weights=model.weights, intercept=model.intercept)
        os.replace(tmp, path / "weights.npz")
