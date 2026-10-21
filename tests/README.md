# Test plan

There are no executable tests yet because algorithms and adapters are not implemented. `python -m pytest` currently exits with code 5 (no tests collected); this is not a passing test suite.

Add tests with each implementation:
- Known-answer RMS/MAV/WL calculations and window boundary cases.
- Custom model predictions, numerical behavior, edge cases, and reference comparisons.
- Adapter shapes, labels/units, timestamps, and target alignment using synthetic fixtures.
- Disjoint subject/session/recording splits and training-only transform fitting.

Keep tests small, deterministic, and independent of downloaded datasets. Mark any future real-data integration checks separately and document how to run them.
