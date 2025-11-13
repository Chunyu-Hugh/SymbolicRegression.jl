# Custom Initial Population Usage Guide

## Overview

SymbolicRegression now supports custom initial populations! You can directly specify the initial population instead of using randomly generated populations.

## How to Call SymbolicRegression.jl

Yes, **SymbolicRegression.jl is Julia code and needs to be called through Julia**. There are several ways:

### Method 1: Direct use in Julia

```julia
using SymbolicRegression

X = randn(Float32, 3, 100)
y = 2.0 * X[1, :] .+ 3.0 * X[2, :] .^ 2

options = Options(; population_size=50, maxsize=10)
hall_of_fame = equation_search(X, y; options=options, niterations=20)
```

### Method 2: Through PySR (Python Interface)

If you use Python, you can call it through PySR:

```python
from pysr import PySRRegressor

model = PySRRegressor(
    population_size=50,
    maxsize=10,
    niterations=20,
)
model.fit(X, y)
```

**Note**: PySR is a Python wrapper that calls Julia code underneath. Currently PySR may not yet support the `initial_population` parameter, so you need to use Julia code directly to use this feature.

### Method 3: Run in Julia Script

Create a `.jl` file, then run:

```bash
julia your_script.jl
```

## Using Custom Initial Population

### Basic Usage

```julia
using SymbolicRegression
using DynamicExpressions: parse_expression

# Prepare data
X = randn(Float32, 3, 100)
y = 2.0 * X[1, :] .+ 3.0 * X[2, :] .^ 2

# Set options
options = Options(;
    binary_operators=[+, *, -, /],
    population_size=50,
    maxsize=10,
)

dataset = Dataset(X, y)

# Create custom population
custom_members = []
for expr_str in ["x1 + x2", "x1 * x2", "x1^2 + x2^2"]
    tree = parse_expression(
        expr_str;
        operators=options.operators,
        variable_names=dataset.variable_names,
        node_type=options.node_type,
    )
    member = PopMember(dataset, tree, options; deterministic=options.deterministic)
    push!(custom_members, member)
end

# Supplement random members to fill population
if length(custom_members) < options.population_size
    random_pop = Population(
        dataset;
        population_size=options.population_size - length(custom_members),
        nlength=3,
        options=options,
        nfeatures=size(X, 1),
    )
    append!(custom_members, random_pop.members)
end

custom_population = Population(custom_members)

# Use custom population
hall_of_fame = equation_search(
    X, y;
    options=options,
    initial_population=custom_population,
    niterations=20,
)
```

### Parameter Description

The `initial_population` parameter can be:

1. **Single Population object**: Used for all outputs and all populations
   ```julia
   initial_population=my_population
   ```

2. **Vector of Populations**: 
   - If length equals number of outputs, each output uses corresponding population
   - If length equals number of populations, each population uses corresponding population
   - Otherwise, all use the first population

3. **nothing** (default): Use randomly generated population

### Important Notes

1. **Population size must match**: The size of the custom population must equal `options.population_size`, otherwise the system will fall back to random population and issue a warning.

2. **Loss will be recalculated**: Even if members in the custom population already have loss values, they will be re-evaluated on the new dataset.

3. **Difference from guesses**:
   - `guesses`: Added to population through migration mechanism (does not replace entire population)
   - `initial_population`: Directly replaces entire initial population

4. **Can be used together**: You can use both `guesses` and `initial_population` together, guesses will be added to population through migration mechanism during search.

## Complete Example

See `examples/custom_initial_population.jl` for a complete example, including:

1. Creating custom population from string expressions
2. Creating custom population from existing search results
3. Manually creating PopMember objects

Run the example:

```bash
julia examples/custom_initial_population.jl
```

## Frequently Asked Questions

### Q: Can I use this feature in Python?

A: Currently PySR (Python interface) may not yet support the `initial_population` parameter. You need to use Julia code directly, or wait for PySR to be updated.

### Q: How to pass custom population from Python to Julia?

A: This is an advanced usage that requires:
1. Create expression strings in Python
2. Pass to Julia through some method (e.g., JSON)
3. Parse and create Population object in Julia

Or, you can complete the entire workflow directly in Julia.

### Q: Can custom population size be smaller than population_size?

A: No. If sizes don't match, the system will fall back to random population. You need to supplement random members to fill the population.

### Q: How to create custom population from previous search results?

A: Extract members from HallOfFame:

```julia
# First search
hof1 = equation_search(X, y; options=options, niterations=10)

# Extract best members
best_members = []
for size in 1:min(5, options.maxsize)
    if hof1.exists[size]
        push!(best_members, hof1.members[size])
    end
end

# Supplement random members
# ... (see example above)

# Create new population and continue search
custom_pop = Population(best_members)
hof2 = equation_search(X, y; options=options, initial_population=custom_pop, niterations=20)
```

## Summary

- ✅ SymbolicRegression.jl is Julia code, needs to be called through Julia
- ✅ Now supports custom initial population through `initial_population` parameter
- ✅ Can be used together with `guesses` parameter
- ⚠️ Population size must match `options.population_size`
- 📝 See `examples/custom_initial_population.jl` for complete example

