# Toy Examples with Code

```julia
using SymbolicRegression
using SymbolicRegression: machine, fit!, predict, report
```

## 1. Simple search

Here's a simple example where we
find the expression `2 cos(x4) + x1^2 - 2`.

```julia
X = 2randn(1000, 5)
y = @. 2*cos(X[:, 4]) + X[:, 1]^2 - 2

model = SRRegressor(
    binary_operators=[+, -, *, /],
    unary_operators=[cos],
    niterations=30
)
mach = machine(model, X, y)
fit!(mach)
```

Let's look at the returned table:

```julia
r = report(mach)
r
```

We can get the selected best tradeoff expression with:

```julia
r.equations[r.best_idx]
```

## 2. Custom operator

Here, we define a custom operator and use it to find an expression:

```julia
X = 2randn(1000, 5)
y = @. 1/X[:, 1]

my_inv(x) = 1/x

model = SRRegressor(
    binary_operators=[+, *],
    unary_operators=[my_inv],
)
mach = machine(model, X, y)
fit!(mach)
r = report(mach)
println(r.equations[r.best_idx])
```

## 3. Multiple outputs

Here, we do the same thing, but with multiple expressions at once,
each requiring a different feature. This means that we need to use
[`MultitargetSRRegressor`](@ref) instead of [`SRRegressor`](@ref):

```julia
X = 2rand(1000, 5) .+ 0.1
y = @. 1/X[:, 1:3]

my_inv(x) = 1/x

model = MultitargetSRRegressor(; binary_operators=[+, *], unary_operators=[my_inv])
mach = machine(model, X, y)
fit!(mach)
```

The report gives us lists of expressions instead:

```julia
r = report(mach)
for i in 1:3
    println("y[$(i)] = ", r.equations[i][r.best_idx[i]])
end
```

## 4. Plotting an expression

For now, let's consider the expressions for output 1 from the previous example:
We can get a SymbolicUtils version with [`node_to_symbolic`](@ref):

```julia
using SymbolicUtils

eqn = node_to_symbolic(r.equations[1][r.best_idx[1]])
```

We can get the LaTeX version with `Latexify`:

```julia
using Latexify

latexify(string(eqn))
```

We can also plot the prediction against the truth:

```julia
using Plots

ypred = predict(mach, X)
scatter(y[1, :], ypred[1, :], xlabel="Truth", ylabel="Prediction")
```

## 5. Other types

SymbolicRegression.jl can handle most numeric types you wish to use.
For example, passing a `Float32` array will result in the search using
32-bit precision everywhere in the codebase:

```julia
X = 2randn(Float32, 1000, 5)
y = @. 2*cos(X[:, 4]) + X[:, 1]^2 - 2

model = SRRegressor(binary_operators=[+, -, *, /], unary_operators=[cos], niterations=30)
mach = machine(model, X, y)
fit!(mach)
```

we can see that the output types are `Float32`:

```julia
r = report(mach)
best = r.equations[r.best_idx]
println(typeof(best))
# Expression{Float32,Node{Float32},...}
```

We can also use `Complex` numbers:

```julia
cos_re(x::Complex{T}) where {T} = cos(abs(x)) + 0im

X = 15 .* rand(ComplexF64, 1000, 5) .- 7.5
y = @. 2*cos_re((2+1im) * X[:, 4]) + 0.1 * X[:, 1]^2 - 2

model = SRRegressor(
    binary_operators=[+, -, *, /],
    unary_operators=[cos_re],
    maxsize=30,
    niterations=100
)
mach = machine(model, X, y)
fit!(mach)
```

## 6. Dimensional constraints

MySRCore accepts explicit dimension metadata as seven-component exponent vectors in
the fixed order `length, mass, time, current, temperature, luminosity, amount`.
The `formula_type` in `Options` selects the contract: `:empirical` ignores dimensions,
`:semi_theoretical` enforces dimensional compatibility inside the expression and fits
an external coefficient `C_dim`, while `:theoretical` also requires the final expression
to have the declared output dimension. There is no dimensional penalty parameter and
no separate dimensionless-constant switch.

For example, for a target proportional to the square of a length variable:

```julia
using MySRCore

X = reshape(Float64[1, 2, 3, 4], 1, :)
y = 3 .* vec(X).^2
length_dim = [1, 0, 0, 0, 0, 0, 0]
area_dim = [2, 0, 0, 0, 0, 0, 0]

options = Options(
    formula_type=:theoretical,
    binary_operators=[+, -, *, /],
    niterations=100,
)
hall = equation_search(
    X,
    y;
    options=options,
    X_dimensions=[length_dim],
    y_dimensions=area_dim,
    parallelism=:serial,
)
```

The same `X_dimensions` and `y_dimensions` arguments are used by the Python MySR
frontend. Invalid candidates are rejected structurally during generation and mutation;
they are not retained with a soft loss penalty.

## 7. Working with Expressions

Expressions in `SymbolicRegression.jl` are represented using the `Expression{T,Node{T},...}` type, which provides a more robust way to combine structure, operators, and constraints. Here's an example:

```julia
using SymbolicRegression

# Define options with operators and structure
options = Options(
    binary_operators=[+, -, *],
    unary_operators=[cos],
)

operators = options.operators
variable_names = ["x1", "x2"]
x1 = Expression(
    Node{Float64}(feature=1),
    operators=operators,
    variable_names=variable_names,
)
x2 = Expression(
    Node{Float64}(feature=2),
    operators=operators,
    variable_names=variable_names,
)

# Construct and evaluate expression
expr = x1 * cos(x2 - 3.2)
X = rand(Float64, 2, 100)
output = expr(X)
```

This `Expression` type, contains the operators used in the expression.
These are what are returned by the search. The raw `Node` type (which is
what used to be output directly) is accessible with

```julia
get_contents(expr)
```

## 8. Template Expressions

Template expressions allow you to define structured expressions where different parts can be constrained to use specific variables.
In this example, we'll create expressions that constrain the functional form in highly specific ways.
(_For more complex examples, see ["Searching with template expressions"](examples/template_expression.md)_ and ["Parameterized Template Expressions"](examples/template_parametric_expression.md)\_)

First, let's set up our basic configuration:

```julia
using SymbolicRegression
using Random: rand, MersenneTwister
using SymbolicRegression: machine, fit!, report
```

The key part is defining our template structure. This determines how different parts of the expression combine:

```julia
expression_spec = @template_spec(expressions=(f, g)) do x1, x2, x3
    f(x1, x2) + g(x2) - g(x3)
end
```

With this structure, we are telling the algorithm that it can learn
any symbolic expressions `f` and `g`, with `f` a function of two inputs,
and `g` a function of one input. The result of

```math
f(x_1, x_2) + g(x_2) - g(x_3)
```

will be compared with the target `y`.

Let's generate some example data:

```julia
n = 100
rng = MersenneTwister(0)
x1 = 10rand(rng, n)
x2 = 10rand(rng, n)
x3 = 10rand(rng, n)
X = (; x1, x2, x3)
y = [
    2 * cos(x1[i] + 3.2) + x2[i]^2 - 0.8 * x3[i]^2
    for i in eachindex(x1)
]
```

Now, remember our structure: for the model to learn this,
it would need to correctly disentangle the contribution
of `f` and `g`!

Now we can set up and train our model by passing the structure in to `expression_spec`:

```julia
model = SRRegressor(;
    binary_operators=(+, -, *, /),
    unary_operators=(cos,),
    niterations=500,
    maxsize=25,
    expression_spec=expression_spec,
)

mach = machine(model, X, y)
fit!(mach)
```

If all goes well, you should see a printout with the following expression:

```text
y = ╭ f = ((#2 * 0.2) * #2) + (cos(#1 + 0.058407) * -2)
    ╰ g = #1 * (#1 * 0.8)
```

This is what we were looking for! We can see that under
$f(x_1, x_2) + g(x_2) - g(x_3)$, this correctly expands to
$2 \cos(x_1 + 3.2) + x_2^2 - 0.8 x_3^2$.

We can also access the individual parts of the template expression
directly from the report:

```julia
r = report(mach)
best_expr = r.equations[r.best_idx]

# Access individual parts of the template expression
println("f: ", get_contents(best_expr).f)
println("g: ", get_contents(best_expr).g)
```

The `TemplateExpression` combines these under the structure
so we can directly and efficiently evaluate this:

```julia
best_expr(randn(3, 20))
```

The above code demonstrates how template expressions can be used to:

- Define structured expressions with multiple components
- Constrains which variables can be used in each component
- Create expressions that can output multiple values

You can even output custom structs - see the more detailed [Template Expression example](examples/template_expression.md)!

Be sure to also check out the [Parametric Template Expressions example](examples/template_parametric_expression.md).

## 9. Logging with TensorBoard

You can track the progress of symbolic regression searches using TensorBoard or other logging backends. Here's an example using `TensorBoardLogger` and wrapping it with [`SRLogger`](@ref):

```julia
using SymbolicRegression
using SymbolicRegression: machine, fit!, report
using TensorBoardLogger

logger = SRLogger(TBLogger("logs/sr_run"))

# Create and fit model with logger
model = SRRegressor(
    binary_operators=[+, -, *],
    maxsize=40,
    niterations=100,
    logger=logger
)

X = (a=rand(500), b=rand(500))
y = @. 2 * cos(X.a * 23.5) - X.b^2

mach = machine(model, X, y)
fit!(mach)
```

You can then view the logs with:

```bash
tensorboard --logdir logs
```

The TensorBoard interface will show
the loss curves over time (at each complexity), as well
as the Pareto frontier volume which can be used as an overall metric
of the search performance.

## 10. Using Differential Operators

`SymbolicRegression.jl` supports differential operators via [`DynamicDiff.jl`](https://github.com/MilesCranmer/DynamicDiff.jl), allowing you to include derivatives directly within template expressions.
Here is an example where we discover the integral of $\frac{1}{x^2 \sqrt{x^2 - 1}}$ in the range $x > 1$.

First, let's generate some data for the integrand:

```julia
using SymbolicRegression
using Random

rng = MersenneTwister(42)
x = 1 .+ rand(rng, 1000) * 9  # Sampling points in the range [1, 10]
y = @. 1 / (x^2 * sqrt(x^2 - 1))  # Values of the integrand
```

Now, define the template for the derivative operator:

```julia
using SymbolicRegression: D

expression_spec = @template_spec(expressions=(f,)) do x
    D(f, 1)(x)
end
```

We can now set up the model to find the symbolic expression for the integral:

```julia
using SymbolicRegression: machine, fit!, report

model = SRRegressor(
    binary_operators=(+, -, *, /),
    unary_operators=(sqrt,),
    maxsize=20,
    expression_spec=expression_spec,
)

X = (; x=x)
mach = machine(model, X, y)
fit!(mach)
```

The learned expression will represent $f(x)$, the indefinite integral of the given function. The derivative of $f(x)$ should match the target $\frac{1}{x^2 \sqrt{x^2 - 1}}$.

You can access the best expression from the report:

```julia
r = report(mach)
best_expr = r.equations[r.best_idx]

println("Learned expression: ", best_expr)
```

If successful, the result should simplify to something like $\frac{\sqrt{x^2 - 1}}{x}$, which is the integral of the target function.

## 11. Seeding search with initial guesses

You can also provide initial guesses for the search.
In this example, let's look for the following function:

```math
\sin(x_1 x_2 + 0.1) + \cos(x_3) x_4 + \frac{x_5}{x_6^2 + 1}
```

```julia
using SymbolicRegression
using SymbolicRegression: machine, fit!, predict, report

X = randn(Float32, 6, 2048)
y = @. sin(X[1, :] * X[2, :] + 0.1f0) + cos(X[3, :]) * X[4, :] + X[5, :] / (X[6, :] * X[6, :] + 1)
```

This expression is quite complex. Now, say that we know most of
the structure, but want to further optimize it. We can provide
a guess for the search:

```julia
model = SRRegressor(
    binary_operators=[+, -, *, /],
    unary_operators=[sin, cos],
    maxsize=35,
    niterations=35,
    guesses=["sin(x1 * x2) + cos(x3) * x4 + x5 / (x6 * x6 + 0.9)", #= can provide additional guesses here =#],
    batching=true,
    batch_size=32,
)

mach = machine(model, X', y)
fit!(mach)
```

If everything goes well, it should optimize the `0.9` to `1.0`,
and also discover the `+ 0.1` term inside the sinusoid, whereas
this might have been difficult to discover as fast from the normal search.

You can also provide multiple guesses. For a template expression,
your guesses should be an array of named tuples, such as
`(; f="cos(#1) + 0.1", g="sin(#2) - 0.9")`.

## 12. Higher-arity operators

You can use operators with more than 2 arguments by passing an `OperatorEnum` explicitly.
This operator allows you to declare arbitrary arities by passing them in a `arity => (op1, op2, ...)` format.

Here's an example using a ternary conditional operator:

```julia
using SymbolicRegression
using SymbolicRegression: machine, fit!, predict, report

scalar_ifelse(a, b, c) = a > 0 ? b : c

X = randn(3, 100)
y = [X[1, i] > 0 ? 2*X[2, i] : X[3, i] for i in 1:100]

model = SRRegressor(
    operators=OperatorEnum(
        1 => (),
        2 => (+, -, *, /),
        3 => (scalar_ifelse,)
    ),
    niterations=35,
)
mach = machine(model, X', y)
fit!(mach)
```

This sort of piecewise logic might be difficult to express with only binary operators.

## 13. Configuring mutation weights

You can adjust how often each type of mutation is applied during the search.
Pass `mutations` with a list of mutation types and their relative weights.
An entry replaces the default weight for that mutation type; anything
you don't list keeps its default.

For example, to enable constant optimization (off by default) and increase
the rate of constant perturbation:

```julia
using SymbolicRegression
using SymbolicRegression: machine, fit!, report

X = 2randn(100, 5)
y = @. 2.7182 * X[:, 1] + 3.1415 * X[:, 2]

model = SRRegressor(
    binary_operators=[+, -, *],
    mutations=[
        OptimizeMutation() => 0.1,
        ConstantMutation() => 0.5,
    ],
    niterations=30,
)
mach = machine(model, X, y)
fit!(mach)
```

See the [`Options`](@ref) documentation for all available mutation types
and their default weights.

## 14. Additional features

For the many other features available in SymbolicRegression.jl,
check out the API page for `Options`. You might also find it useful
to browse the documentation for the Python frontend
[PySR](http://astroautomata.com/PySR), which has additional documentation.
In particular, the [tuning page](http://astroautomata.com/PySR/tuning)
is useful for improving search performance.
