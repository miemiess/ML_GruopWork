# Experiment configurations

Copy `experiment.example.toml` to a named versioned config after auditing a dataset. The example is a specification template; no configuration loader or training runner exists yet.

Use paths relative to the repository root. Keep machine-specific overrides in `*.local.toml` (ignored). Never put credentials or private absolute paths in a tracked configuration.

Record exact split identities separately and link their file from the completed config. A seed alone is insufficient to recreate a split after the dataset changes.
