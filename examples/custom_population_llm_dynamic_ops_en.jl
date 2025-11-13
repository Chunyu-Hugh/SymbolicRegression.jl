"""
Custom Initial Population + LLM Iterative Optimization Example (Dynamic Operator Extension)

This example extends the basic workflow from `custom_population_llm.jl` by adding logic for
"dynamically expanding the operator library by round", demonstrating:

1. Initial search only carries basic operators;
2. After each round of interaction with the LLM, extract new operators that appear in candidate expressions;
3. Add new operators to the operator library and continue searching with the expanded configuration.

If you don't put all operators into `Options` in advance, you just need to add them when constructing
new `Options` and custom populations in the next round. The only prerequisite is: when new operators
are detected, you need to prepare Julia function implementations for them and register them in the
operator mapping table.

To run this example, you need:
- A properly configured local SymbolicRegression.jl development environment
- Access to a working LLM API (this example uses the same configuration as `examples/llm_integration.jl`)
- HTTP and JSON packages installed (`Pkg.add(["HTTP", "JSON"])`)
"""

using SymbolicRegression
using SymbolicRegression:
    equation_search,
    calculate_pareto_frontier,
    compute_complexity,
    string_tree,
    Dataset,
    Population,
    PopMember,
    Options,
    safe_pow
using SymbolicRegression.MutationFunctionsModule: mutate_constant, mutate_feature, mutate_operator
using SymbolicRegression.CoreModule: create_expression
using DynamicExpressions: parse_expression, with_type_parameters

using HTTP
using JSON

# ===========================================
# LLM API Configuration (directly uses examples/llm_integration.jl)
# ===========================================
const MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
const API_BASE_URL = "https://ollama.cs.odu.edu"
const API_KEY = "sk-704f0fe8734549c9b57ac09a69001ad7"

function call_llm(prompt::String; timeout::Int=120, max_retries::Int=3)
    url = "$API_BASE_URL/api/v1/chat/completions"
    headers = Dict(
        "Content-Type" => "application/json",
        "Authorization" => "Bearer $API_KEY",
    )
    payload = Dict(
        "model" => MODEL_NAME,
        "messages" => [Dict("role" => "user", "content" => prompt)],
        "stream" => false,
    )

    last_error = nothing
    for attempt in 1:max_retries
        try
            response = HTTP.post(
                url,
                headers=headers,
                body=JSON.json(payload);
                readtimeout=timeout,
                connecttimeout=30,
            )
            if response.status == 200
                body_str = String(response.body)
                result = JSON.parse(body_str)
                if haskey(result, "choices") && !isempty(result["choices"])
                    return result["choices"][1]["message"]["content"]
                else
                    return "Error: No choices in response - $(body_str[1:min(200, length(body_str))])"
                end
            else
                error_msg =
                    "Error: $(response.status) - $(String(response.body)[1:min(200, length(String(response.body)))])"
                if attempt < max_retries
                    @warn "API call failed (attempt $(attempt)/$(max_retries)): $(error_msg), retrying..."
                    sleep(2)
                    continue
                else
                    return error_msg
                end
            end
        catch e
            last_error = e
            if attempt < max_retries
                if isa(e, HTTP.Exceptions.TimeoutError)
                    @warn "Request timeout (attempt $(attempt)/$(max_retries)), retrying... (timeout: $(timeout)s)"
                else
                    @warn "Request failed (attempt $(attempt)/$(max_retries)): $(e), retrying..."
                end
                sleep(2)
            else
                if isa(e, HTTP.Exceptions.TimeoutError)
                    return "Request failed: TimeoutError after $(max_retries) attempts (timeout: $(timeout)s)."
                else
                    return "Request failed after $(max_retries) attempts: $(e)"
                end
            end
        end
    end
    return "Request failed: $(last_error)"
end

function format_equations(dominating::AbstractVector{<:PopMember}, options)
    lines = String[]
    push!(lines, "Pareto optimal expressions (sorted by complexity ascending):")
    push!(lines, "Complexity\tLoss\tExpression")
    push!(lines, "-" ^ 80)
    for member in dominating
        complexity = compute_complexity(member, options)
        loss = member.loss
        equation = string_tree(member.tree, options)
        push!(lines, "$complexity\t$loss\t$equation")
    end
    return join(lines, "\n")
end

function create_llm_prompt(
    equations_text::String,
    round::Int,
    total_rounds::Int,
    select_count::Int,
    population_size::Int,
    current_binary_ops::Vector{String},
    current_unary_ops::Vector{String},
)
    new_needed = population_size - select_count
    
    # List all available operators (including those in the library but not activated)
    available_unary = join(keys(UNARY_OPERATOR_LIBRARY), ", ")
    available_binary = join(keys(BINARY_OPERATOR_LIBRARY), ", ")
    
    current_ops_info = "Currently activated operators:\n"
    current_ops_info *= "  Binary operators: $(join(current_binary_ops, ", "))\n"
    current_ops_info *= "  Unary operators: $(join(current_unary_ops, ", "))\n"
    current_ops_info *= "\nSystem-supported operator library:\n"
    current_ops_info *= "  Available unary operators: $available_unary\n"
    current_ops_info *= "  Available binary operators: $available_binary\n"
    
    return """
You are a symbolic regression and physical modeling expert. We have completed round $round/$total_rounds of search.

$current_ops_info

$equations_text

Please complete the following tasks:
1. Analyze the physical meaning, types (e.g., polynomial, trigonometric, rational functions, etc.), and the physical mechanisms these expressions might describe.
2. Select $select_count expressions that are most worth keeping and exploring further (indices start from 1).
3. **Important: Based on your analysis, suggest new operators that need to be added**. If the current operators are insufficient to express patterns in the data, please explicitly suggest which new operators are needed (choose from the system-supported operator library).
4. Based on your analysis, generate $new_needed new candidate expressions with the following requirements:
   
   - You can use currently activated operators
   - If you suggested new operators, you can use them in expressions (even if they are not yet activated)
   - Complexity should be comparable to or slightly higher than the selected expressions
   - Expressions must use variable names x1, x2, x3, x4, x5
   - No variable names other than x1, x2, x3, x4, x5 are allowed

Please return in JSON format:
{
  "analysis": "...",
  "selected_indices": [1, 3, 5],
  "suggested_operators": {
    "binary": ["max", "min"],
    "unary": ["cos", "sin"]
  },
  "new_expressions": [
    "2.0 * cos(x4) + safe_pow(x1, 2) - 2.0",
    "(x1 + x2) * cos(x4)",
    "safe_pow(x1 + x2, 2) + 0.5 * x5"
  ],
  "reasoning": "..."
}

Note:
- Operators in suggested_operators must come from the system-supported operator library
- If no new operators are needed, suggested_operators can be empty arrays
- Suggested operators will be automatically activated before the next round of search
"""
end

function parse_llm_response(response::String)
    json_start = findfirst('{', response)
    json_end = findlast('}', response)
    if json_start === nothing || json_end === nothing
        chars = collect(response)
        snippet = String(chars[1:min(length(chars), 200)])
        return (Int[], String[], String[], String[], "Failed to parse JSON. Response snippet: $snippet")
    end
    parsed = try
        JSON.parse(response[json_start:json_end])
    catch e
        return (Int[], String[], String[], String[], "JSON parsing failed: $e")
    end

    indices = if haskey(parsed, "selected_indices") && parsed["selected_indices"] isa Vector
        [Int(i) for i in parsed["selected_indices"] if i isa Integer || occursin(r"^\d+$", string(i))]
    else
        Int[]
    end

    new_exprs = if haskey(parsed, "new_expressions") && parsed["new_expressions"] isa Vector
        [String(expr) for expr in parsed["new_expressions"] if expr isa AbstractString]
    else
        String[]
    end

    # Extract suggested new operators
    suggested_binary = String[]
    suggested_unary = String[]
    if haskey(parsed, "suggested_operators") && parsed["suggested_operators"] isa Dict
        ops = parsed["suggested_operators"]
        if haskey(ops, "binary") && ops["binary"] isa Vector
            suggested_binary = [String(op) for op in ops["binary"] if op isa AbstractString]
        end
        if haskey(ops, "unary") && ops["unary"] isa Vector
            suggested_unary = [String(op) for op in ops["unary"] if op isa AbstractString]
        end
    end

    analysis = if haskey(parsed, "analysis")
        val = parsed["analysis"]
        if val isa AbstractString
            val
        elseif val isa Dict || val isa Vector
            JSON.json(val)
        else
            string(val)
        end
    else
        ""
    end
    return (indices, new_exprs, suggested_binary, suggested_unary, analysis)
end

function sanitize_expression(expr::String)
    sanitized = replace(expr, r"\bpow\s*\(" => "safe_pow(")

    pow_pattern = r"(?<!safe_pow\()(\([^\(\)]+\)|[a-zA-Z_][a-zA-Z0-9_\.]*)\s*\^\s*(-?\d+(?:\.\d+)?)"
    while true
        m = match(pow_pattern, sanitized)
        m === nothing && break
        base = strip(m.captures[1])
        power = strip(m.captures[2])
        replacement = "safe_pow($base, $power)"
        sanitized = replace(sanitized, m.match => replacement; count=1)
    end

    return sanitized
end

function parse_expressions_to_members(
    expr_strings::Vector{String},
    dataset::Dataset{T,L},
    options::Options,
    nfeatures::Int,
    template_members::AbstractVector{<:PopMember},
) where {T,L}
    isempty(template_members) && error("template_members must not be empty")
    members = PopMember[]
    deterministic_bool = options.deterministic isa Bool ? options.deterministic : false
    node_type_with_T = with_type_parameters(options.node_type, T)
    for expr_str in expr_strings
        sanitized = sanitize_expression(expr_str)
        try
            tree = parse_expression(
                sanitized;
                operators=options.operators,
                variable_names=dataset.variable_names,
                node_type=node_type_with_T,
                expression_type=options.expression_type,
            )
            member = PopMember(dataset, tree, options; deterministic=deterministic_bool)
            push!(members, member)
        catch e
            error_msg = sprint(showerror, e, catch_backtrace())
            @warn "Failed to parse expression '$expr_str' (sanitized='$sanitized'): $error_msg"
        end
    end
    return members
end

function create_manual_population(dataset::Dataset, options::Options)
    temp_pop = Population(
        dataset;
        population_size=1,
        nlength=3,
        options=options,
        nfeatures=size(dataset.X, 1),
    )
    manual_members = typeof(temp_pop.members[1])[]

    contains_feature(tree) = any(tree) do node
        node.degree == 0 && !node.constant
    end

    while length(manual_members) < options.population_size
        tree = gen_random_tree(3, options, size(dataset.X, 1), Float64)
        contains_feature(tree) || continue
        member = PopMember(dataset, tree, options; deterministic=options.deterministic)
        push!(manual_members, member)
    end
    standardized = [
        PopMember(dataset, member.tree.tree, options; deterministic=options.deterministic) for
        member in manual_members
    ]
    return Population(standardized), standardized
end

function clone_member(member::PopMember, dataset::Dataset, options::Options)
    deterministic_bool = options.deterministic isa Bool ? options.deterministic : false
    return PopMember(dataset, member.tree.tree, options; deterministic=deterministic_bool)
end

function mutate_member(
    base_member::PopMember,
    dataset::Dataset,
    options::Options,
    nfeatures::Int,
)::PopMember
    mutated_tree = copy(base_member.tree.tree)
    mutation_ops = [:constant, :operator, :feature]
    for _ in 1:5
        op = rand(mutation_ops)
        try
            if op == :constant
                mutated_tree = mutate_constant(mutated_tree, 0.25, options)
            elseif op == :operator
                mutated_tree = mutate_operator(mutated_tree, options)
            else
                mutated_tree = mutate_feature(mutated_tree, nfeatures)
            end
            break
        catch
            continue
        end
    end
    return PopMember(dataset, mutated_tree, options; deterministic=options.deterministic)
end

function ensure_member_count!(
    target_vector::Vector{T},
    target_count::Int,
    source_pool::AbstractVector{<:PopMember},
    dataset::Dataset,
    options::Options,
    nfeatures::Int,
) where {T<:PopMember}
    idx = 1
    if isempty(source_pool)
        error("ensure_member_count! requires a non-empty source_pool to generate members.")
    end
    while length(target_vector) < target_count
        base = source_pool[((idx - 1) % length(source_pool)) + 1]
        push!(target_vector, mutate_member(base, dataset, options, nfeatures))
        idx += 1
    end
    length(target_vector) > target_count && resize!(target_vector, target_count)
end

function rebuild_population(
    selected_members::AbstractVector{<:PopMember},
    new_members::AbstractVector{<:PopMember},
    fallback_members::AbstractVector{<:PopMember},
    dataset::Dataset,
    options::Options,
    target_selected::Int,
)
    nfeatures = size(dataset.X, 1)
    member_type = eltype(fallback_members)
    selected_block = Vector{member_type}()
    for member in selected_members
        push!(selected_block, clone_member(member, dataset, options))
    end
    ensure_member_count!(
        selected_block,
        target_selected,
        isempty(selected_members) ? fallback_members : selected_members,
        dataset,
        options,
        nfeatures,
    )

    target_new = options.population_size - target_selected
    new_block = Vector{member_type}()
    for member in new_members
        push!(new_block, clone_member(member, dataset, options))
    end
    ensure_member_count!(
        new_block,
        target_new,
        isempty(new_members) ? fallback_members : new_members,
        dataset,
        options,
        nfeatures,
    )

    combined = vcat(selected_block, new_block)
    return Population(combined)
end

# ===========================================
# Dynamic Operator Management
# ===========================================

const UNARY_OPERATOR_LIBRARY = Dict(
    "cos" => cos,
    "sin" => sin,
    "tan" => tan,
    "exp" => exp,
    "log" => log,
    "sqrt" => sqrt,
    "abs" => abs,
)

const BINARY_OPERATOR_LIBRARY = Dict(
    "+" => +,
    "-" => -,
    "*" => *,
    "/" => /,
    "safe_pow" => safe_pow,
    "max" => max,
    "min" => min,
)

mutable struct OperatorState
    binary_names::Vector{String}
    binary_funcs::Vector{Function}
    unary_names::Vector{String}
    unary_funcs::Vector{Function}
end

function OperatorState(; binary::Vector{String}=["+", "-", "*", "/", "safe_pow"], unary::Vector{String}=String[])
    binary_funcs = Function[]
    for name in binary
        if haskey(BINARY_OPERATOR_LIBRARY, name)
            push!(binary_funcs, BINARY_OPERATOR_LIBRARY[name])
        else
            error("Binary operator '$name' is not registered in BINARY_OPERATOR_LIBRARY.")
        end
    end
    unary_funcs = Function[]
    for name in unary
        if haskey(UNARY_OPERATOR_LIBRARY, name)
            push!(unary_funcs, UNARY_OPERATOR_LIBRARY[name])
        else
            error("Unary operator '$name' is not registered in UNARY_OPERATOR_LIBRARY.")
        end
    end
    return OperatorState(copy(binary), binary_funcs, copy(unary), unary_funcs)
end

function operator_names_from_exprs(exprs::Vector{String})
    binary_needed = String[]
    unary_needed = String[]
    for expr in exprs
        sanitized = sanitize_expression(expr)
        for name in keys(UNARY_OPERATOR_LIBRARY)
            occursin(Regex("\\b$(name)\\s*\\("), sanitized) || continue
            push!(unary_needed, name)
        end
        for name in keys(BINARY_OPERATOR_LIBRARY)
            if !isletter(first(name))
                continue
            end
            occursin(Regex("\\b$(name)\\s*\\("), sanitized) || continue
            push!(binary_needed, name)
        end
        occursin(r"safe_pow\s*\(", sanitized) && push!(binary_needed, "safe_pow")
    end
    return (unique(binary_needed), unique(unary_needed))
end

function operator_names_from_members(members::Vector{<:PopMember}, options::Options)
    binary_needed = String[]
    unary_needed = String[]
    for member in members
        expr_str = string_tree(member.tree, options)
        binary, unary = operator_names_from_exprs([expr_str])
        append!(binary_needed, binary)
        append!(unary_needed, unary)
    end
    return (unique(binary_needed), unique(unary_needed))
end

function update_operator_state!(
    state::OperatorState,
    required_binary::Vector{String},
    required_unary::Vector{String},
)
    added_binary = String[]
    added_unary = String[]

    for name in required_binary
        name in state.binary_names && continue
        if haskey(BINARY_OPERATOR_LIBRARY, name)
            push!(state.binary_names, name)
            push!(state.binary_funcs, BINARY_OPERATOR_LIBRARY[name])
            push!(added_binary, name)
        else
            @warn "Detected unregistered binary operator '$name', please add its Julia implementation to BINARY_OPERATOR_LIBRARY."
        end
    end

    for name in required_unary
        name in state.unary_names && continue
        if haskey(UNARY_OPERATOR_LIBRARY, name)
            push!(state.unary_names, name)
            push!(state.unary_funcs, UNARY_OPERATOR_LIBRARY[name])
            push!(added_unary, name)
        else
            @warn "Detected unregistered unary operator '$name', please add its Julia implementation to UNARY_OPERATOR_LIBRARY."
        end
    end

    return (added_binary=added_binary, added_unary=added_unary)
end

function build_options_from_state(
    state::OperatorState;
    population_size::Int,
    maxsize::Int,
    verbosity::Int,
)
    return Options(;
        binary_operators=copy(state.binary_funcs),
        unary_operators=copy(state.unary_funcs),
        population_size=population_size,
        maxsize=maxsize,
        verbosity=verbosity,
    )
end

# ===========================================
# Main Workflow
# ===========================================

function run_llm_guided_search_dynamic_ops(;
    num_rounds::Int=3,
    iterations_per_round::Int=20,
    select_count::Int=5,
    population_size::Int=20,
    maxsize::Int=14,
    verbosity::Int=0,
)
    println("=" ^ 80)
    println("Custom Initial Population + LLM Iterative Optimization Example (Dynamic Operator Extension)")
    println("=" ^ 80)

    X = randn(Float64, 5, 100)
    y = 2 .* cos.(X[4, :]) .+ X[1, :].^2 .- 2
    y = y .+ randn(100) .* 1e-3

    operator_state = OperatorState()
    options = build_options_from_state(
        operator_state;
        population_size=population_size,
        maxsize=maxsize,
        verbosity=verbosity,
    )

    dataset = Dataset(X, y; variable_names=["x1", "x2", "x3", "x4", "x5"])

    println("Creating initial manual population for first round...")
    baseline_population, manual_members = create_manual_population(dataset, options)
    println("First round population member count: $(length(baseline_population.members))")

    current_population = baseline_population

    for round in 1:num_rounds
        println("\n" * "=" ^ 80)
        println("Starting round $round/$num_rounds of search")
        println("=" ^ 80)

        hof = equation_search(
            X, y;
            options=options,
            initial_population=current_population,
            parallelism=:serial,
            niterations=iterations_per_round,
        )

        dominating = calculate_pareto_frontier(hof)
        println("\nThis round's Pareto frontier expressions:")
        for member in dominating
            complexity = compute_complexity(member, options)
            loss = member.loss
            equation = string_tree(member.tree, options)
            println("Complexity: $complexity | Loss: $loss | Expression: $equation")
        end

        # Note: equation_search itself does not generate new operators, it can only use operators defined in Options
        # This detection is defensive programming; in practice, equation_search results will only contain defined operators
        # The real source of new operators is expressions returned by LLM (see new_exprs processing below)
        if !isempty(dominating)
            required_binary_from_search, required_unary_from_search = operator_names_from_members(dominating, options)
            updates_from_search = update_operator_state!(operator_state, required_binary_from_search, required_unary_from_search)
            if !isempty(updates_from_search.added_binary) || !isempty(updates_from_search.added_unary)
                println("\nDetected new operators from search results, operator library expanded:")
                !isempty(updates_from_search.added_binary) &&
                    println("  New binary operators: $(join(updates_from_search.added_binary, ", "))")
                !isempty(updates_from_search.added_unary) &&
                    println("  New unary operators: $(join(updates_from_search.added_unary, ", "))")
                
                options = build_options_from_state(
                    operator_state;
                    population_size=population_size,
                    maxsize=maxsize,
                    verbosity=verbosity,
                )
                baseline_population, manual_members = create_manual_population(dataset, options)
            end
        end

        round == num_rounds && break

        if isempty(dominating)
            println("Warning: No Pareto frontier expressions found this round, using latest manual population for next round.")
            current_population = baseline_population
            continue
        end

        select_count = min(select_count, length(dominating))
        prompt = create_llm_prompt(
            format_equations(dominating, options),
            round,
            num_rounds,
            select_count,
            options.population_size,
            operator_state.binary_names,
            operator_state.unary_names,
        )

        println("\nSending request to LLM...")
        response = call_llm(prompt)
        println("LLM response:\n$response\n")

        selected_indices, new_exprs, suggested_binary, suggested_unary, analysis = parse_llm_response(response)
        println("LLM analysis: $analysis")
        
        # First handle LLM's explicit suggestions for new operators (before parsing expressions)
        if !isempty(suggested_binary) || !isempty(suggested_unary)
            println("\nLLM suggested adding new operators:")
            !isempty(suggested_binary) &&
                println("  Suggested binary operators: $(join(suggested_binary, ", "))")
            !isempty(suggested_unary) &&
                println("  Suggested unary operators: $(join(suggested_unary, ", "))")
            
            # Validate and add suggested operators
            updates_from_suggestion = update_operator_state!(operator_state, suggested_binary, suggested_unary)
            if !isempty(updates_from_suggestion.added_binary) || !isempty(updates_from_suggestion.added_unary)
                println("Added LLM-suggested new operators:")
                !isempty(updates_from_suggestion.added_binary) &&
                    println("  New binary operators: $(join(updates_from_suggestion.added_binary, ", "))")
                !isempty(updates_from_suggestion.added_unary) &&
                    println("  New unary operators: $(join(updates_from_suggestion.added_unary, ", "))")
                
                # Update options and population
                options = build_options_from_state(
                    operator_state;
                    population_size=population_size,
                    maxsize=maxsize,
                    verbosity=verbosity,
                )
                baseline_population, manual_members = create_manual_population(dataset, options)
                println("Operator library updated, next round will use expanded operator set.")
            else
                println("Note: Suggested operators may already be in the operator library, or not in the system-supported operator library.")
            end
        end
        if isempty(selected_indices)
            println("LLM did not return valid indices, defaulting to first $select_count expressions.")
            selected_indices = collect(1:select_count)
        else
            selected_indices = [
                idx for idx in selected_indices if 1 <= idx <= length(dominating)
            ]
            isempty(selected_indices) && (selected_indices = collect(1:select_count))
        end

        selected_members_old = [dominating[idx] for idx in selected_indices]
        println("Selected expression count: $(length(selected_members_old)) -> indices: $(selected_indices)")

        selected_expr_strings = [
            string_tree(member.tree, options) for member in selected_members_old
        ]
        all_candidate_exprs = vcat(selected_expr_strings, new_exprs)
        # Detect new operators from LLM-returned expressions (new_exprs)
        # This is the main source of new operators: LLM can "suggest" expressions containing new operators
        required_binary, required_unary = operator_names_from_exprs(all_candidate_exprs)

        updates = update_operator_state!(operator_state, required_binary, required_unary)
        if !isempty(updates.added_binary) || !isempty(updates.added_unary)
            println("Detected new operators from LLM-returned expressions, operator library expanded:")
            !isempty(updates.added_binary) &&
                println("  New binary operators: $(join(updates.added_binary, ", "))")
            !isempty(updates.added_unary) &&
                println("  New unary operators: $(join(updates.added_unary, ", "))")

            options = build_options_from_state(
                operator_state;
                population_size=population_size,
                maxsize=maxsize,
                verbosity=verbosity,
            )
            baseline_population, manual_members = create_manual_population(dataset, options)
        end

        println(
            "Current operator set: binary=$(join(operator_state.binary_names, ", ")), unary=$(join(operator_state.unary_names, ", "))",
        )

        selected_members = parse_expressions_to_members(
            selected_expr_strings,
            dataset,
            options,
            size(X, 1),
            manual_members,
        )
        println("Successfully rebuilt selected expression count: $(length(selected_members))")

        expected_new = options.population_size - length(selected_members)
        new_members = parse_expressions_to_members(
            new_exprs,
            dataset,
            options,
            size(X, 1),
            manual_members,
        )
        println("Successfully parsed LLM new expression count: $(length(new_members)) (expected: $expected_new)")

        current_population = rebuild_population(
            selected_members,
            new_members,
            manual_members,
            dataset,
            options,
            length(selected_members),
        )

        println("Building next round custom population completed.")
    end

    println("\nSearch completed.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_llm_guided_search_dynamic_ops(; num_rounds=3, iterations_per_round=20, select_count=5)
end

