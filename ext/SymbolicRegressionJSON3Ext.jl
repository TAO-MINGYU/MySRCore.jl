# Modified from SymbolicRegression.jl 2.0.0-beta.8 for the MySRCore namespace.
module SymbolicRegressionJSON3Ext

using JSON3: JSON3
import MySRCore.SymbolicRegression.UtilsModule: json3_write

function json3_write(trace, tracing_file; append::Bool)
    open(tracing_file, append ? "a" : "w") do io
        JSON3.write(io, trace; allow_inf=true)
        write(io, '\n')
    end
end

end
