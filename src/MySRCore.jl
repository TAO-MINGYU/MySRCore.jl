module MySRCore

# MySRCore wraps the retained upstream module so its public package identity can
# evolve independently without obscuring provenance or making upstream merges noisy.
using Reexport

include("SymbolicRegression.jl")
@reexport using .SymbolicRegression

export SymbolicRegression

end
