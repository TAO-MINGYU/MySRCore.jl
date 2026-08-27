# MySRCore.jl

MySRCore.jl is the Julia algorithm core for MySR, a general-purpose symbolic
regression project. The package currently establishes a rename-safe baseline around
SymbolicRegression.jl 2.0.0-beta.8; MySR-specific algorithm changes will be added
incrementally with focused tests.

## Installation

MySR installs this package automatically through JuliaPkg. Direct Julia users can
install the same pinned release from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/TAO-MINGYU/MySRCore.jl", rev="v0.1.0")
using MySRCore
```

## Development setup

```julia
using Pkg
Pkg.develop(path="/home/taomingyu/projects/MySRCore.jl")
using MySRCore
```

`MySRCore` re-exports the retained upstream API, including `Options`,
`equation_search`, and `calculate_pareto_frontier`.

## Source provenance

This repository is derived from
[SymbolicRegression.jl](https://github.com/astroautomata/SymbolicRegression.jl),
version 2.0.0-beta.8 at commit
`35d45fd625dc8df0067df60c72b615d83518ed44`. See [NOTICE](NOTICE),
[VENDORING.md](VENDORING.md), [FORK_CHANGES.md](FORK_CHANGES.md), and
[UPSTREAM_README.md](UPSTREAM_README.md).

MySRCore.jl is an independent fork and is not an official SymbolicRegression.jl
release.

## License

Licensed under Apache License 2.0. Upstream copyright and attribution are retained.
