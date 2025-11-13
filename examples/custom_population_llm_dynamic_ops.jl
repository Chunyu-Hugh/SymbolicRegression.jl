"""
自定义初始种群 + LLM 迭代优化示例（动态算子扩展版）

在基础示例 `custom_population_llm.jl` 的流程上，新增了「按轮次动态扩展算子库」的逻辑，用于演示：

1. 初始搜索只携带基础算子；
2. 每轮与 LLM 交互后，提取候选表达式中出现的新算子；
3. 将新算子加入算子库，并基于扩展后的配置继续搜索。

如果不提前把所有算子放进 `Options`，只要在下一轮构造新 `Options` 和自定义种群时补齐即可。唯一的前提是：当检测到新的算子时，需要为它们准备 Julia 函数实现，并登记在算子映射表中。

要运行本示例，需要：
- 已正确设置本地 SymbolicRegression.jl 开发环境
- 具备可用的 LLM API（示例沿用 `examples/llm_integration.jl` 中的配置）
- 已安装 HTTP 和 JSON（`Pkg.add(["HTTP", "JSON"])`）
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
# LLM API 配置（直接沿用 examples/llm_integration.jl）
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
                    @warn "API调用失败 (尝试 $(attempt)/$(max_retries)): $(error_msg)，正在重试..."
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
                    @warn "请求超时 (尝试 $(attempt)/$(max_retries))，正在重试... (超时时间: $(timeout)秒)"
                else
                    @warn "请求失败 (尝试 $(attempt)/$(max_retries)): $(e)，正在重试..."
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
    push!(lines, "Pareto 最优表达式（按复杂度升序）:")
    push!(lines, "复杂度\t损失(Loss)\t表达式")
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
你是一位符号回归与物理建模专家。我们已经完成第 $round/$total_rounds 轮搜索。

$equations_text

请完成以下任务：
1. 分析这些表达式的物理含义、类型（例如多项式、三角函数、有理函数等），以及它们可能描述的物理机制。
2. 选出 $select_count 个最值得保留和继续探索的表达式（索引从 1 开始）。
3. 基于你的分析，再生成 $new_needed 个新的候选表达式，要求：
   
   - 使用 +, -, *, /, safe_pow(·,·) 等运算
   - 如需引入其他算子，请直接写出函数名，如 cos、sin、exp、log 等
   - 复杂度与被选中的表达式相当或略高
   - 表达式必须使用变量名 x1,x2,x3,x4,x5
   - 不能出现除 x1,x2,x3,x4,x5 以外的变量名

请以 JSON 返回，格式：
{
  "analysis": "...",
  "selected_indices": [1, 3, 5],
  "new_expressions": [
    "2.0 * cos(x4) + safe_pow(x1, 2) - 2.0",
    "(x1 + x2) * cos(x4)",
    "safe_pow(x1 + x2, 2) + 0.5 * x5"
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
        return (Int[], String[], "无法解析 JSON。响应片段: $snippet")
    end
    parsed = try
        JSON.parse(response[json_start:json_end])
    catch e
        return (Int[], String[], "JSON 解析失败: $e")
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
            @warn "无法解析表达式 '$expr_str' (sanitized='$sanitized'): $error_msg"
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
# 动态算子管理
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

function OperatorState(; binary::Vector{String}=["+", "-", "*", "/", "safe_pow"], unary::Vector{String}=["cos"])
    binary_funcs = Function[
        get(
            BINARY_OPERATOR_LIBRARY,
            name,
            error("二元算子 '$name' 未在 BINARY_OPERATOR_LIBRARY 中注册。"),
        ) for name in binary
    ]
    unary_funcs = Function[
        get(
            UNARY_OPERATOR_LIBRARY,
            name,
            error("一元算子 '$name' 未在 UNARY_OPERATOR_LIBRARY 中注册。"),
        ) for name in unary
    ]
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
            @warn "检测到未注册的二元算子 '$name'，请在 BINARY_OPERATOR_LIBRARY 中补充其 Julia 实现。"
        end
    end

    for name in required_unary
        name in state.unary_names && continue
        if haskey(UNARY_OPERATOR_LIBRARY, name)
            push!(state.unary_names, name)
            push!(state.unary_funcs, UNARY_OPERATOR_LIBRARY[name])
            push!(added_unary, name)
        else
            @warn "检测到未注册的一元算子 '$name'，请在 UNARY_OPERATOR_LIBRARY 中补充其 Julia 实现。"
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
# 主流程
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
    println("自定义初始种群 + LLM 迭代优化示例（动态算子扩展版）")
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

    println("创建首轮手动初始种群...")
    baseline_population, manual_members = create_manual_population(dataset, options)
    println("首轮种群成员数: $(length(baseline_population.members))")

    current_population = baseline_population

    for round in 1:num_rounds
        println("\n" * "=" ^ 80)
        println("开始第 $round/$num_rounds 轮搜索")
        println("=" ^ 80)

        hof = equation_search(
            X, y;
            options=options,
            initial_population=current_population,
            parallelism=:serial,
            niterations=iterations_per_round,
        )

        dominating = calculate_pareto_frontier(hof)
        println("\n本轮 Pareto 前沿表达式:")
        for member in dominating
            complexity = compute_complexity(member, options)
            loss = member.loss
            equation = string_tree(member.tree, options)
            println("复杂度: $complexity | 损失: $loss | 表达式: $equation")
        end

        round == num_rounds && break

        if isempty(dominating)
            println("警告: 本轮未找到 Pareto 前沿表达式，下一轮使用最新手动种群。")
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
        )

        println("\n向 LLM 发送请求...")
        response = call_llm(prompt)
        println("LLM 响应:\n$response\n")

        selected_indices, new_exprs, analysis = parse_llm_response(response)
        println("LLM 分析: $analysis")
        if isempty(selected_indices)
            println("LLM 未返回有效索引，默认选择前 $select_count 个表达式。")
            selected_indices = collect(1:select_count)
        else
            selected_indices = [
                idx for idx in selected_indices if 1 <= idx <= length(dominating)
            ]
            isempty(selected_indices) && (selected_indices = collect(1:select_count))
        end

        selected_members_old = [dominating[idx] for idx in selected_indices]
        println("选中的表达式数量: $(length(selected_members_old)) -> 索引: $(selected_indices)")

        selected_expr_strings = [
            string_tree(member.tree, options) for member in selected_members_old
        ]
        all_candidate_exprs = vcat(selected_expr_strings, new_exprs)
        required_binary, required_unary = operator_names_from_exprs(all_candidate_exprs)

        updates = update_operator_state!(operator_state, required_binary, required_unary)
        if !isempty(updates.added_binary) || !isempty(updates.added_unary)
            println("检测到新的算子，已扩展算子库：")
            !isempty(updates.added_binary) &&
                println("  新增二元算子: $(join(updates.added_binary, \", \"))")
            !isempty(updates.added_unary) &&
                println("  新增一元算子: $(join(updates.added_unary, \", \"))")

            options = build_options_from_state(
                operator_state;
                population_size=population_size,
                maxsize=maxsize,
                verbosity=verbosity,
            )
            baseline_population, manual_members = create_manual_population(dataset, options)
        end

        println(
            "当前算子集合: 二元=$(join(operator_state.binary_names, \", \")), 一元=$(join(operator_state.unary_names, \", \"))",
        )

        selected_members = parse_expressions_to_members(
            selected_expr_strings,
            dataset,
            options,
            size(X, 1),
            manual_members,
        )
        println("成功重建选中表达式数量: $(length(selected_members))")

        expected_new = options.population_size - length(selected_members)
        new_members = parse_expressions_to_members(
            new_exprs,
            dataset,
            options,
            size(X, 1),
            manual_members,
        )
        println("成功解析 LLM 新表达式数量: $(length(new_members)) (期望: $expected_new)")

        current_population = rebuild_population(
            selected_members,
            new_members,
            manual_members,
            dataset,
            options,
            length(selected_members),
        )

        println("构建下一轮自定义种群完成。")
    end

    println("\n搜索结束。")
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_llm_guided_search_dynamic_ops(; num_rounds=3, iterations_per_round=20, select_count=5)
end


