# Fork changes

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
