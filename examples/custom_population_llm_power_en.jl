"""
Custom Initial Population + LLM Iterative Optimization Example (Power Function Formula)

Workflow:
1. Use method 3 (manually create PopMember) to generate the first batch of custom population,
   and use it as the initial population for equation_search.
2. After completing one round of search, extract results from the Pareto frontier and call LLM:
   - Analyze the physical meaning/category of candidate expressions
   - Select several promising expressions
   - Generate new expressions based on the analysis
3. Reconstruct the LLM-selected + generated expressions into a new custom population,
   and continue the next round of search.
4. Repeat the above process for multiple rounds (num_rounds is configurable).

To run this example, you need:
- A properly configured local SymbolicRegression.jl development environment
- Access to a working LLM API (this example uses the same configuration as llm_integration.jl)
- HTTP and JSON packages installed (`Pkg.add(["HTTP", "JSON"])`)

Formula: y = X^alpha * (1-X)^beta, where alpha=0.1, beta=-0.5
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
    safe_pow,
    gen_random_tree
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
)
    new_needed = population_size - select_count
    return """
You are a symbolic regression and physical modeling expert. We have completed round $round/$total_rounds of search.

$equations_text

Please complete the following tasks:
1. Analyze the physical meaning, types (e.g., polynomial, power function, rational function, etc.), and the physical mechanisms these expressions might describe.
2. Select $select_count expressions that are most worth keeping and exploring further (indices start from 1).
3. Based on your analysis, generate $new_needed new candidate expressions with the following requirements:
   - Use variable x1
   - Use operations such as +, -, *, /, safe_pow(·,·)
   - Complexity should be comparable to or slightly higher than the selected expressions
   - Reflect your judgment about physical meaning

Please return in JSON format:
{
  "analysis": "...",
  "selected_indices": [1, 3, 5],
  "new_expressions": [
    "safe_pow(x1, 0.1) * safe_pow(1 - x1, -0.5)",
    "safe_pow(x1, 0.2) * safe_pow(1 - x1, -0.4)",
    "safe_pow(x1 + 0.1, 0.1) * safe_pow(1 - x1, -0.5)"
  ],
  "reasoning": "..."
}
"""
end

function parse_llm_response(response::String)
    json_start = findfirst('{', response)
    json_end = findlast('}', response)
    if json_start === nothing || json_end === nothing
        chars = collect(response)
        snippet = String(chars[1:min(length(chars), 200)])
        return (Int[], String[], "Failed to parse JSON. Response snippet: $snippet")
    end
    parsed = try
        JSON.parse(response[json_start:json_end])
    catch e
        return (Int[], String[], "JSON parsing failed: $e")
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
    return (indices, new_exprs, analysis)
end

function sanitize_expression(expr::String)
    # LLM-returned expressions should already be in x1 format, just need to convert ^ to safe_pow
    sanitized = expr
    
    # Convert all ^ operators to safe_pow, but skip those already in safe_pow(...)
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
    # Ensure node_type contains the correct data type
    node_type_with_T = with_type_parameters(options.node_type, T)
    for expr_str in expr_strings
        sanitized = sanitize_expression(expr_str)
        try
            # Parse expression string into tree structure, using correct data type
            tree = parse_expression(
                sanitized;
                operators=options.operators,
                variable_names=dataset.variable_names,
                node_type=node_type_with_T,
                expression_type=options.expression_type,
            )
            # Directly use PopMember constructor, which internally calls create_expression
            # If create_expression fails, this will throw an error
            member = PopMember(dataset, tree, options; deterministic=deterministic_bool)
            push!(members, member)
        catch e
            # Provide more detailed error information
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

function run_llm_guided_search(; num_rounds::Int=3, iterations_per_round::Int=20, select_count::Int=5)
    println("=" ^ 80)
    println("Custom Initial Population + LLM Iterative Optimization Example (Power Function Formula)")
    println("=" ^ 80)

    # 1. Data and options
    # Generate 1000 samples, X in [0, 1] range
    n_samples = 1000
    X = rand(Float64, 1, n_samples)  # 1x1000, values in [0, 1]
    alpha = 0.1
    beta = -0.5
    # y = X^alpha * (1-X)^beta
    y = (X .^ alpha) .* ((1 .- X) .^ beta)
    # Convert to vector and add random noise
    y = vec(y) .+ randn(n_samples) .* 1e-3

    options = Options(;
        binary_operators=[+, *, -, /, safe_pow],
        unary_operators=[],  # Remove cos, as the new formula doesn't need it
        population_size=20,
        maxsize=14,
        verbosity=0,
    )

    dataset = Dataset(X, y; variable_names=["x1"])

    println("Creating manual initial population (method 3)...")
    custom_population, manual_members = create_manual_population(dataset, options)
    println("First round population member count: $(length(custom_population.members))")

    current_population = custom_population

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

        round == num_rounds && break

        if isempty(dominating)
            println("Warning: No Pareto frontier expressions found this round, using manual backup population for next round.")
            current_population = custom_population
            continue
        end

        select_count = min(select_count, length(dominating))
        prompt = create_llm_prompt(
            format_equations(dominating, options),
            round,
            num_rounds,
            select_count,
            options.population_size,
        )

        println("\nSending request to LLM...")
        response = call_llm(prompt)
        println("LLM response:\n$response\n")

        selected_indices, new_exprs, analysis = parse_llm_response(response)
        println("LLM analysis: $analysis")
        if isempty(selected_indices)
            println("LLM did not return valid indices, defaulting to first $select_count expressions.")
            selected_indices = collect(1:select_count)
        else
            selected_indices = [
                idx for idx in selected_indices if 1 <= idx <= length(dominating)
            ]
            isempty(selected_indices) && (selected_indices = collect(1:select_count))
        end

        selected_members = [dominating[idx] for idx in selected_indices]
        println("Selected expression count: $(length(selected_members)) -> indices: $(selected_indices)")

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
    run_llm_guided_search(; num_rounds=3, iterations_per_round=20, select_count=5)
end

