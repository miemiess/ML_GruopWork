# Data acquisition and provenance

Data is not bundled or downloaded automatically. Start with a small, legally accessible sample from each original source, inspect its schema, and record feasibility before a full download.

## Sources

| Local dataset ID | Source | Acquisition and constraints |
| --- | --- | --- |
| `uci_gestures` | [UCI EMG Data for Gestures](https://archive.ics.uci.edu/dataset/481/emg+data), DOI 10.24432/C5ZP5C | Download the original archive from UCI. CC BY 4.0; retain attribution. Preserve participant folders and recording files. |
| `putemg_force` | [Provider page](https://biolab.put.poznan.pl/putemg-dataset/) and [official downloader](https://github.com/biolab-put/putemg-downloader) | Select `emg_force` and CSV or HDF5, initially for a few available participant IDs. Video/depth are unnecessary for this planned EMG-only study. Confirm terms for the exact force release before use or redistribution. |
| `zenodo_fatigue` | [Zenodo v2, record 14182446](https://zenodo.org/records/14182446), DOI 10.5281/zenodo.14182446 | Obtain EMG, self-perceived-fatigue ratings, and protocol files; the EMG archive is about 3.3 GB. Check the release license/terms before use and record them. Keep anthropometric metadata local. |

## Layout

- `raw/<dataset_id>/`: original archives/extracted source files, unchanged.
- `interim/<dataset_id>/`: cleaned and time-aligned recordings.
- `processed/<dataset_id>/`: window features, targets, and grouping/split metadata.
- `manifests/`: reviewed provenance records, checksums, and exclusion decisions without sensitive participant details.

The first three folders are ignored by Git. Do not add payloads with `git add -f`. Manifests are tracked intentionally; review them before commit. Compute a source checksum with PowerShell `Get-FileHash -Algorithm SHA256 path/to/file` or the equivalent on your platform.

## Dataset-specific audit

**UCI:** eight channels plus time and class columns. Label 0 is unmarked, not the rest class; rest is label 1. Choose and document whether to exclude unmarked samples and incomplete-class recordings. Class 7 is not present for every participant. Preserve recording and participant IDs.

**putEMG-Force:** use measured `FORCE_1 ... FORCE_10` as candidate regression targets, never the requested `TRAJ_*` trajectory as a substitute. Choose outputs explicitly and confirm physical meaning, units, calibration/MVC handling, synchronization, and missing trials. The provider describes the combined collection as 44 participants across two acquisitions; audit the force subset rather than assuming every participant has complete force recordings.

**Zenodo fatigue:** eight muscles from 13 healthy adults. The release describes EMG at 1259 Hz and ratings at 50 Hz, with levels 0/1/2. Align by timestamps, define the label assigned to each window, and inspect missing/ambiguous intervals. These are self-perceived labels, not clinical ground truth.

Do not pool tasks or equate channel indices across datasets. Validate file-level facts against downloaded data. Report actual retained participants/trials/classes and exclusions after auditing.

## Manifest checklist

Copy `manifests/manifest.example.toml` per dataset and record source URL/version, retrieval date, license evidence, relative file paths, SHA-256 hashes, formats, units, channel maps, group IDs, label definitions, and exclusions. Keep split assignments versioned alongside the experiment. Until these checks are complete, data availability and task scope remain provisional.
