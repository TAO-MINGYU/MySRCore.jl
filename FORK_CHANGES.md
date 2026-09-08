# Fork changes

## 2026-09-08 - MySRCore 1.1.2

- Published the patch release paired with MySR 1.1.2 for the corrected benchmark
  scoring and four-group capability ablation protocol.

## 2026-09-02 - MySRCore 1.1.0

- Added the backend half of formula_type-conditioned RNN-GPSR: dimensional-aware
  structural seeding, lightweight GPSR feedback, and configurable round budgets.
- Added batched proposal validation and formal-population injection that keeps
  validated user guesses ahead of RNN-GPSR and random seeds.
- Added focused regression tests for dimensional generation, feedback, budgets,
  non-finite costs, and user-guess priority.

## 2026-09-01 - MySRCore 1.0.0

- Promoted the MySRCore package to its first stable release for the MySR 1.0.0
  frontend/backend contract.
- Added strict and compatible dimension policies selected by formula_type,
  including dimension-aware feature metadata and the semi-theoretical C_dim
  boundary behavior.
- Removed the inherited soft dimensional-constraint APIs; callers now provide
  explicit X_dimensions and y_dimensions metadata.

## 2026-08-27 - MySRCore 0.1.0 release foundation

- Added a pinned upstream baseline document and explicit modified-file notices.
- Distinguished the MySRCore changelog and citation guidance from the retained
  SymbolicRegression.jl history.
- Documented GitHub-tag installation for MySR and direct Julia users.

## 2026-08-26 - MySRCore 0.1.0 foundation

The following files differ from the retained SymbolicRegression.jl baseline:

- `Project.toml`: changed the public package name, UUID, version, and authors.
- `src/MySRCore.jl`: added the MySRCore public wrapper module.
- `ext/*.jl`: routed optional extensions through the MySRCore package boundary.
- `README.md`, `NOTICE`, and release metadata: identified this independent fork and
  preserved upstream attribution.
- Tests: added MySRCore package-contract and public-API checks.
- `test/runtests.jl`: added a minimal MySRCore package-identity test; the unchanged
  upstream suite is retained under `upstream_test/` for later migration.

The algorithm implementation under `src/SymbolicRegression.jl` and its included
source files remains the attributed upstream baseline unless a later entry names a
specific modification.
## 2026-09-07 - MySRCore 1.1.1

- Versioned the backend release paired with the MySR 1.1.1 RNN-GPSR fix.
