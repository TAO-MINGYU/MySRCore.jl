# Upstream source baseline

MySRCore.jl is an independent derivative of
[SymbolicRegression.jl](https://github.com/astroautomata/SymbolicRegression.jl).

| Field | Value |
| --- | --- |
| Upstream version | `2.0.0-beta.8` |
| Upstream commit | `35d45fd625dc8df0067df60c72b615d83518ed44` |
| Upstream license | Apache License 2.0 |
| MySRCore package UUID | `65629769-74ff-4293-a07c-ae7483a03f6e` |

The retained algorithm implementation is the nested
`MySRCore.SymbolicRegression` module under `src/SymbolicRegression.jl` and its
included source files. MySRCore-specific package identity, namespace routing,
release metadata, and later algorithm changes are recorded in
[FORK_CHANGES.md](FORK_CHANGES.md).

Future upstream updates must be reviewed and merged against the pinned commit.
Do not replace the MySRCore source tree wholesale, because doing so would discard
the independent package namespace and MySR-specific changes.
