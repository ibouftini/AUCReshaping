# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

The library's code has not changed.

### Changed
- The repository is renamed `AUCReshaping`, after the paper. The package is still `aucreshape`.
- The evaluation is being redone under a stricter protocol, published before the new runs. The
  Results page now says so, and lists the corrections to the first evaluation.

### Added
- Documentation: "The implementation and the paper", one row for each choice the paper leaves open.
- The evaluation code, in `benchmarks/`. It is not part of the installed package.

### Fixed
- Status text that described the project as complete.

## [0.1.0] - 2026-11-03

A tested, documented PyTorch implementation of AUCReshaping (Bhat et al., *Scientific Reports*
13, 21097, 2023), and an evaluation of when it helps. The package is installed from GitHub; it
is not published on PyPI. The project is complete and not under active development.

### Added
- `aucreshape.functional`: `boost_values`, `reshaping_weights`, `auc_reshaping_loss`
  (`prob`/`logit`/`weight` modes, `band`/`sum`/`max` combine rules, masks, `pos_weight`,
  `weight`) and `linear_ramp` boost schedules.
- `aucreshape.metrics`: `threshold_at_specificity`, `sensitivity_at_threshold`,
  `sensitivity_at_specificity` and `partial_auc`, tested against scikit-learn.
- `AUCReshapingLoss`, a drop-in replacement for `BCEWithLogitsLoss`.
- `ThresholdTracker` with `fixed`, `batch`, `window`, `ema` and `epoch` strategies,
  DDP-aware, checkpointable, with `calibrate`/`from_scores` for validation thresholds.
- `aucreshape.integrations.lightning.AUCReshapingCallback` (the `lightning` extra).
- Examples for plain PyTorch, Lightning, Hugging Face `Trainer`, DDP, custom per-sample losses,
  multi-label with missing labels, checkpoint/resume and validation thresholds.
- A one-time warning when a training-mode loss is discarded without `backward()` (a
  validation pass with gradients enabled would otherwise update the tracker silently).
- A documentation site, including a Results page: benchmarks on six public datasets, a
  synthetic laboratory of eight scenarios with controls, and an end-to-end NIH ChestX-ray14
  study, with where AUCReshaping may help and where it should not be used.
