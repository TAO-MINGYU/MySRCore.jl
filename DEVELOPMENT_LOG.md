# MySRCore mutation-affinity worktree log

## 2026-09-05 — Decision/Implementation

- Scope: isolated worktree `feature/dimension-aware-mutation-affinity-dev`; the
  original MySRCore checkout remains available to an active benchmark.
- Confirmed: point mutation now filters operator and feature destinations using
  the existing `formula_type` dimension policy before sampling.
- Decision: use a static operator-family prior (`+/-`, `*/`, `sin/cos`,
  `sinh/cosh`) with uniform exploration; no historical transition learning is
  included in this milestone.
- Changed paths: `src/MutationAffinity.jl`, `src/Core.jl`, `src/Options.jl`,
  `src/OptionsStruct.jl`, `src/MutationFunctions.jl`, `src/Mutate.jl`, and
  `test/runtests.jl`.
- Verification: direct test runner passed all testsets; package-level `Pkg.test()`
  passed; a one-iteration strict-dimensional `equation_search` smoke run
  completed and returned four members.
- Residual risk: default strength `4.0` and exploration `0.2` are tunable
  hypotheses and need an eventual ablation/benchmark; no history-based learning
  or performance claim is made here.
