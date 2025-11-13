"""
自定义初始种群 + LLM 迭代优化示例（幂函数公式）

流程：
1. 使用方法3（手动创建 PopMember）生成首批自定义种群，并作为 equation_search 的初始种群。
2. 完成一轮搜索后，从 Pareto 前沿提取结果，调用 LLM：
   - 分析候选表达式的物理含义/类别
   - 选择若干有潜力的表达式
   - 基于分析生成新的表达式
3. 将 LLM 选择+生成的表达式重新构造成新的自定义种群，继续下一轮搜索。
4. 重复上述过程多轮（num_rounds 可配置）。

要运行本示例，需要：
- 已正确设置本地 SymbolicRegression.jl 开发环境
- 具备可用的 LLM API（示例使用 llm_integration.jl 中同样的配置）
- 已安装 HTTP 和 JSON（`Pkg.add(["HTTP", "JSON"])`）

公式：y = X^alpha * (1-X)^beta，其中 alpha=0.1, beta=-0.5
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
1. 分析这些表达式的物理含义、类型（例如多项式、幂函数、有理函数等），以及它们可能描述的物理机制。
2. 选出 $select_count 个最值得保留和继续探索的表达式（索引从 1 开始）。
3. 基于你的分析，再生成 $new_needed 个新的候选表达式，要求：
   - 使用变量 x1
   - 使用 +, -, *, /, safe_pow(·,·) 等运算
   - 复杂度与被选中的表达式相当或略高
   - 体现你对物理含义的判断

请以 JSON 返回，格式：
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
    # LLM 返回的表达式应该已经是 x1 等格式，只需要将 ^ 转换为 safe_pow
    sanitized = expr
    
    # 将所有的 ^ 运算符转换为 safe_pow，但跳过已经在 safe_pow(...) 中的
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
    # 确保 node_type 包含正确的数据类型
    node_type_with_T = with_type_parameters(options.node_type, T)
    for expr_str in expr_strings
        sanitized = sanitize_expression(expr_str)
        try
            # 解析表达式字符串为树结构，使用正确的数据类型
            tree = parse_expression(
                sanitized;
                operators=options.operators,
                variable_names=dataset.variable_names,
                node_type=node_type_with_T,
                expression_type=options.expression_type,
            )
            # 直接使用 PopMember 构造函数，它会内部调用 create_expression
            # 如果 create_expression 失败，这里会抛出错误
            member = PopMember(dataset, tree, options; deterministic=deterministic_bool)
            push!(members, member)
        catch e
            # 提供更详细的错误信息
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

function run_llm_guided_search(; num_rounds::Int=3, iterations_per_round::Int=20, select_count::Int=5)
    println("=" ^ 80)
    println("自定义初始种群 + LLM 迭代优化示例（幂函数公式）")
    println("=" ^ 80)

    # 1. 数据与选项
    # 生成 1000 个样本，X 在 [0, 1] 范围内
    n_samples = 1000
    X = rand(Float64, 1, n_samples)  # 1x1000，值在 [0, 1]
    alpha = 0.1
    beta = -0.5
    # y = X^alpha * (1-X)^beta
    y = (X .^ alpha) .* ((1 .- X) .^ beta)
    # 转换为向量并添加随机噪声
    y = vec(y) .+ randn(n_samples) .* 1e-3

    options = Options(;
        binary_operators=[+, *, -, /, safe_pow],
        unary_operators=[],  # 移除 cos，因为新公式不需要
        population_size=20,
        maxsize=14,
        verbosity=0,
    )

    dataset = Dataset(X, y; variable_names=["x1"])

    println("创建手动初始种群（方法3）...")
    custom_population, manual_members = create_manual_population(dataset, options)
    println("首轮种群成员数: $(length(custom_population.members))")

    current_population = custom_population

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
            println("警告: 本轮未找到 Pareto 前沿表达式，下一轮使用手动备份种群。")
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

        selected_members = [dominating[idx] for idx in selected_indices]
        println("选中的表达式数量: $(length(selected_members)) -> 索引: $(selected_indices)")

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
    run_llm_guided_search(; num_rounds=3, iterations_per_round=20, select_count=5)
end

