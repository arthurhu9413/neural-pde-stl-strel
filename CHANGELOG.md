# Changelog

All notable changes to this repository are documented here.

The project follows a pragmatic "artifact-first" policy: metrics and figures
should come from committed scripts and result files, not from hand-edited prose.

## [0.7.9] -- 2026-03-28

- Fixed minimum-version enforcement in clean installs that do not have the separate `packaging` distribution installed.
- Added an internal lightweight version-comparison fallback used by the CLI doctor gate, `require_optional()`, and monitor backend guards.
- Added regression coverage for minimum-version checks without `packaging`, including a clean-install style fallback path.

## [0.7.8] -- 2026-03-28

### Fixed

- Made the CLI `doctor` requirement gate version-aware for dependencies with declared support floors, so too-old installations of NumPy / PyYAML (and other probed minimum-version dependencies) no longer count as healthy merely because they are importable.
- Added regression coverage for the too-old dependency case in both the pytest suite and the standalone test runner.

## [0.7.7] -- 2026-03-28

### Fixed

- Made `python -m neural_pde_stl_strel doctor` validate the core runtime requirements by default, so dependency-free installs now fail fast instead of incorrectly reporting a healthy environment.
- Clarified the optional-install documentation to distinguish packaged framework extras from manual installs, and added the missing `spatial-spec` / SpaTiaL installation paths.
- Removed a stale code comment that incorrectly described `frameworks/` as an implicit namespace package even though it is a regular subpackage.

## [0.7.6] -- 2026-03-28

### Fixed

- Realigned the CLI dependency-group semantics with the package metadata and install docs: `doctor --require core` now tracks the true base runtime set (`numpy`, `pyyaml`) instead of incorrectly treating PyTorch as a core requirement.
- Made unqualified `doctor --require core` and `doctor --require all` default to the intuitive `all` policy while preserving the existing `any` default for the `physics` and `stl` groups.
- Corrected the `scripts/check_env.py` exit-code documentation to mention the PyYAML requirement and tightened the Makefile help text so `install-extra` now accurately describes the `.[all]` extra as the set of extras declared in `pyproject.toml`.

## [0.7.5] -- 2026-03-28

### Fixed

- Added PyYAML to both environment-check surfaces (`python -m neural_pde_stl_strel about/doctor` and `scripts/check_env.py`) so the repository now probes all declared base runtime dependencies instead of silently omitting YAML support.
- Extended the CLI requirement-group logic and regression tests so the `core` dependency checks now cover PyYAML alongside the existing base probes.

## [0.7.4] -- 2026-03-28

### Fixed

- Aligned the default Ruff configuration with the repository's actual fatal-error lint gate, so plain `ruff check` now matches CI and `make lint` instead of advertising a broader rule set the repo does not yet satisfy.
- Made the CLI reject contradictory output-format flag combinations such as `--brief --json` and `--version --json` instead of silently ignoring one of the requested modes.
- Renamed the CI lint job from `Lint & type check` to `Lint` so the workflow label matches what the job actually runs.

## [0.7.3] -- 2026-03-28

### Fixed

- Modernized the packaging metadata to use an SPDX license string plus explicit `license-files`, removing the current setuptools deprecation warnings during wheel/sdist builds.
- Added a `MANIFEST.in` so source distributions now ship the repo-level reproducibility artifacts (`docs/`, `configs/`, `scripts/`, `results/`, `assets/`, `figs/`, CI files, and support metadata) instead of only the importable package and tests.
- Added regression coverage that checks both the new SPDX-style metadata and the completeness of the built source distribution.

## [0.7.2] -- 2026-03-28

### Fixed

- Replaced the self-referential `all` extra in `pyproject.toml` with an explicit dependency union so wheel metadata no longer points `all` back at the package itself.
- Added explicit setuptools package-data rules so local `__pycache__` / `.pyc` files no longer leak into built wheels after running tests or imports.
- Removed the last plain-text traces of previously removed collaborator/affiliation tokens from the project-identity regression tests while preserving the guardrails.
- Hardened the identity scans in both test suites to ignore generated cache/build metadata directories such as `*.egg-info`, `.ruff_cache`, and `.figure-check`.

## [0.7.1] -- 2026-03-28

### Fixed

- Restored the documented `python -m neural_pde_stl_strel --about` and `--version` CLI aliases so the reproducibility docs and installed command-line behavior match again.
- Added regression tests that lock in the top-level CLI aliases alongside the existing subcommand coverage.
- Removed a few remaining context-specific helper-script/doc wording nits so the repository reads more consistently as a standalone research artifact.

## [0.7.0] -- 2026-03-28

### Changed

- Recast the manuscript-planning material into venue-agnostic `docs/PUBLICATION_NOTES.md` and `docs/MANUSCRIPT_OUTLINE.md` so the repository no longer carries stale venue-specific scaffolding.
- Simplified package, license, and citation metadata to the solo-authored project identity requested for this repository.
- Tightened README wording so the repo now reads as a standalone research artifact instead of a course- or collaborator-specific release bundle.

### Fixed

- Removed stale collaborator names and venue references from docs, tests, packaging metadata, and changelog text.
- Kept version metadata synchronized across `pyproject.toml`, `src/neural_pde_stl_strel/__init__.py`, and `CITATION.cff` after the authorship cleanup.

## [0.6.0] -- 2026-03-18

### Added

- Added `CITATION.cff` so the repository now ships machine-readable citation metadata alongside the publication-facing docs.
- Added project-identity regression tests that lock in the `neural-pde-stl-strel` package/repository name and detect stale legacy identifiers in UTF-8 source/docs files.

### Changed

- Aligned the README title, package metadata, CLI/help text, Makefile banner, and user-facing script descriptions around the `neural-pde-stl-strel` repository identity.
- Tightened the repo and paper-planning docs to use clearer `neural PDE` / `physics-ML` terminology where the old project nickname was unnecessarily ambiguous.
- Updated the C-MAPSS planning notes to reflect mixed current official availability signals rather than treating dataset access as fully settled.
- Clarified in the publication-planning notes that the manuscript-planning guidance should stay concise and evidence-driven rather than target an arbitrary page count.

### Fixed

- Removed the remaining stale course/project-name references from top-level packaging, helper scripts, and repo-facing documentation.
- Kept version metadata synchronized across `pyproject.toml`, `src/neural_pde_stl_strel/__init__.py`, and the new citation file.

## [0.5.5] -- 2026-03-17

### Added

- Added `docs/CLAIMS_AND_EVIDENCE.md` to map safe repo- and paper-level claims to the exact committed artifacts that support them.
- Added `docs/PUBLICATION_NOTES.md` to align the repository with the then-current publication framing, page-budget guidance, and case-study positioning.
- Added `docs/VERIFIED_REFERENCE_POOL.md`, a checked reference pool plus a clearly separated expansion list for manuscript work.
- Added regression tests that lock in the new paper-support docs, the current publication-planning constraints, and the refreshed C-MAPSS wording.

### Changed

- Updated the README, dataset recommendations, reproducibility guide, implementation map, paper-positioning notes, and reading list so time-sensitive venue and dataset guidance matches current official sources.
- Reframed the aerospace-benchmark note around a now-accessible C-MAPSS side branch and a cleaner aerothermal / thermal-transfer PDE path.
- Recorded the committed environment snapshot more explicitly so the repo can support a concrete experimental-setup paragraph.

### Fixed

- Removed the stale claim that NASA C-MAPSS was unavailable for direct download.
- Switched framework-documentation links to the current official PhysicsNeMo docs.
- Avoided PyTorch scalar-conversion warnings in the diffusion and heat training progress logs by detaching tensors before formatting them.

## [0.5.4] -- 2026-03-07

### Fixed

- Switched the GitHub Actions PyTorch job to an explicit Linux CPU-only torch install path before repo extras, which avoids the heavyweight default CUDA wheel selection on hosted Linux runners.
- Added explicit GitHub Actions `permissions: contents: read` and `cache-dependency-path` settings so workflow caching and token scope match the repository's actual dependency layout.
- Added a documented `make install-cpu-linux` path plus matching README / install-guide notes for CPU-only Linux and CI environments.

### Added

- Regression tests that lock in the CPU-only CI install path, explicit workflow permissions, and the Linux CPU install Makefile target.

## [0.5.3] -- 2026-03-07

### Fixed

- Extended repository hygiene rules so repo-local runtime directories from `.env.example` defaults (`.cache/`, `.mplconfig/`, `.pycache/`, `logs/`, `runs/`, `.tmp/`, `data/`, and `.env`) are ignored and cleaned, which prevents accidental cache/config leakage in handoff zips.
- Corrected the stale `.env.example` install note that had implied the Makefile already passed an explicit torch wheel index.
- Normalized heat-rollout metadata path serialization to POSIX form by writing `Path.as_posix()` in `scripts/gen_heat2d_frames.py`, and updated the committed `assets/heat2d_scalar/meta.json` outdir accordingly.

### Added

- Regression tests that lock in the repo-hygiene ignore/clean rules and the portable heat-metadata path style.

## [0.5.2] -- 2026-03-07

### Fixed

- Unified the MoonLight evaluation path with the repository's committed heat-monitoring artifact name, so `scripts/eval_heat2d_moonlight.py` now writes `results/heat2d_strel_monitoring.json` by default instead of a stray alternate filename.
- Corrected the case-sensitive `docs/FRAMEWORK_SURVEY.md` example path in `scripts/framework_survey.py`, avoiding accidental creation of a duplicate lowercase file on Linux/macOS filesystems.
- Cleaned up a stale `scripts/check_env.py` docstring note that incorrectly implied `make doctor` writes `results/hardware.json`.
- Standardized the lone `regularization` spelling in `configs/heat2d_baseline.yaml` to `regularization` for consistency with the rest of the repository.

### Added

- Regression tests that lock in the heat-monitoring JSON filename and the framework-survey docs path.

## [0.5.1] -- 2026-03-07

### Added

- `make refresh-check` plus regression tests that lock in the new semantic
  summary-validation path.

### Changed

- Reworked `scripts/refresh_committed_summaries.py` so the default path reads
  committed dense-field sidecars and refreshes only the text summaries when they
  meaningfully drift.
- Updated the README, reproducibility guide, implementation map, and workflow
  docs to distinguish the stable artifact path from the explicit
  checkpoint-rebuild path.

### Fixed

- Repaired the GitHub Actions summary-refresh step so it validates committed
  `results/` artifacts semantically instead of rewriting tracked binaries and
  diffing them byte-for-byte.
- Stopped the default summary refresh path from rewriting the main diffusion
  `*_field.pt` / `*.npz` sidecars, which was the direct cause of the flaky CI
  failure on tiny floating-point drift.

## [0.5.0] -- 2026-03-07

### Added

- Scratch-space figure validation support in `scripts/generate_all_figures.py`
  via `--output-root`, `--manifest`, and `--check`, so figure generation can be
  exercised in CI without rewriting tracked repository PNGs.
- `make figures-check` and regression tests that lock in the non-destructive
  figure-generation path.

### Changed

- Reframed the README and paper-planning docs around the actual technical
  problem: using STL/STREL to audit and bias field-valued physical-AI models
  while reporting the fidelity/runtime tradeoff honestly.
- Rewrote the framework survey, dataset recommendations, reproducibility guide,
  reading list, and paper-positioning notes to be more technical and to keep
  citation / benchmark guidance grounded in primary sources and official project
  docs.

### Fixed

- Repaired the GitHub Actions figure job so it validates generated artifacts in
  scratch space instead of assuming Matplotlib will reproduce byte-identical PNG
  files across environments.
- Removed the recurring figure-layout warnings in the two multi-panel plots that
  used shared colorbars.

## [0.4.3] -- 2026-03-07

### Changed

- Routed the diffusion `Makefile` targets and the matching README / reproducibility
  commands through `scripts/run_experiment.py`, so the published YAML-driven
  workflow now matches the actual command surface.
- Made the main YAML configs explicit about `experiment` and `tag`, improving
  run-directory names, stable artifact aliases, and dry-run output.

### Fixed

- Repaired the broken `make demo` / `make diffusion1d-demo` path, which had been
  passing unsupported `--config` / `--override` flags to
  `scripts/train_diffusion_stl.py`.
- Taught `scripts/run_experiment.py` and the `heat2d` experiment to honor
  `io.tag` consistently, so config-driven runs now emit descriptive artifact
  names instead of generic `run` placeholders.
- Refreshed the committed pytest summary and quality-dashboard figure after the
  post-fix regression pass.

## [0.4.2] -- 2026-03-06

### Added

- `docs/IMPLEMENTATION_MAP.md` to map repo files to the actual evidence behind
  publication-facing claims.
- `docs/EXAMPLE_STATUS.md` and `scripts/example_status.py` to separate reported
  results, runnable demos, probes, and planned integrations.
- `docs/PAPER_POSITIONING.md` to keep the repository aligned with claims that
  the committed artifacts can actually support.
- `configs/heat2d_stl_safe.yaml` and `configs/heat2d_stl_eventually.yaml` as
  scalar-STL stepping-stone configs for the 2D heat experiment.
- `make status`, `make heat2d-stl-safe`, and `make heat2d-stl-eventually`
  convenience targets.

### Changed

- Expanded the README quickstart, documentation index, and 2D command section to
  make the example inventory and next-step configs easier to find.
- Bumped package metadata to `0.4.2` after the final audit-and-polish pass.

### Fixed

- Repaired the standalone test suite so it no longer hard-codes the previous
  package version.
- Replaced the broken `make benchmark` target with a committed-snapshot viewer
  backed by `results/benchmark_training.csv`.
- Tightened `.gitignore` so timestamped local run directories under `results/`
  are less likely to leak into repository artifacts.
- Refreshed the committed test-summary sidecar after the post-audit checks.

## [0.4.1] -- 2026-03-06

### Changed

- Replaced stale diffusion summary tables with values recomputed directly from
  the committed checkpoints.
- Rebuilt `scripts/generate_all_figures.py` around committed JSON/CSV/field
  artifacts so that key figures are data-driven rather than hand-waved.
- Aligned the default committed STL config with the main reported checkpoint
  (`stl.weight = 5.0`).
- Rewrote the main documentation files to state clearly what is implemented,
  what is monitoring-only, and what is still a framework probe.
- Simplified the reading list to a shorter set of verified references and tool
  links.
- Tightened contributor guidance around citations, summary tables, and paper
  writing.

### Added

- `scripts/refresh_committed_summaries.py` to rebuild summary JSON/CSV files
  from committed checkpoints.
- Committed `.npz` field exports for the main diffusion checkpoints to support
  figure generation without requiring PyTorch at figure-build time.

### Fixed

- Diffusion RTAMT sidecar JSON files now reference existing field artifacts and
  the correct committed time step.
- Benchmark naming now matches the default committed STL run more closely.
- The figure pipeline now labels conceptual diagrams as conceptual and keeps
  publication-facing quantitative plots tied to actual result files.
- Plotting dependencies now include `pandas`, which both
  `refresh_committed_summaries.py` and `generate_all_figures.py` require.
- The `stl` extra and `requirements-extra.txt` now gate RTAMT to Python < 3.12,
  matching the current upstream import limitation on newer Python versions.
- CI now checks that refreshed summaries and regenerated figures do not rewrite
  tracked artifacts.
- Figure-side ablation summaries are now tested against the committed result
  table so drift is caught before release.
- The quality-dashboard figure now reads the nested pytest summary format
  correctly.

## [0.4.0] -- 2026-02-20

### Added

- STREL monitoring support and a 2D heat case study with committed figures and
  monitoring summaries
- paper-planning notes, figure generation, and benchmark tables for the
  early-2026 manuscript push
- expanded publication metadata for an earlier manuscript iteration

### Changed

- README and docs expanded from a course repo into a publication-facing research repo
- figure generation, cost reporting, and framework notes broadened beyond the
  original diffusion-only experiment

## [0.3.0] -- 2026-02-18

### Added

- CLI diagnostics (`python -m neural_pde_stl_strel --about`, `doctor`, `pip`)
- optional-dependency probes for RTAMT, MoonLight, Neuromancer, TorchPhysics,
  and PhysicsNeMo
- environment reporting and benchmark scaffolding
- standalone tests for dependency probing and docs sanity

### Changed

- package metadata and docs polished for broader collaboration
- figures and readme examples expanded

### Fixed

- optional import handling for missing third-party packages
- CI and test stability around lightweight installs

## [0.2.0] -- 2026-02-19

### Added

- publication-quality figure generation from committed result artifacts
- `docs/MANUSCRIPT_OUTLINE.md`
- `docs/SPECIFICATIONS.md`
- contributor onboarding and more explicit project documentation

### Changed

- related-work and documentation coverage expanded
- Makefile and CI checks improved

### Fixed

- figure glyph issues and assorted documentation rough edges

## [0.1.0] -- 2026-02-18

### Added

- core PINN training pipeline for 1D diffusion and 2D heat equations
- differentiable STL penalty functions and smooth temporal operators
- RTAMT integration for offline STL robustness monitoring
- MoonLight integration for spatial-temporal monitoring
- framework probes for Neuromancer, PhysicsNeMo, and TorchPhysics
- lambda ablation support
- standalone and pytest-based test suites
- CI, YAML configs, CSV logging, and environment diagnostics
