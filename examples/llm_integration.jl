"""
SymbolicRegression + LLM 集成示例

这个示例展示了如何将大语言模型（LLM）与SymbolicRegression结合：
1. 进行一轮搜索
2. 将结果发送给LLM
3. LLM根据复杂度、loss、score、equation选择表达式
4. 用选择的表达式创建自定义种群
5. 继续搜索
6. 循环直到达到最大迭代次数
"""

# 需要安装 HTTP 和 JSON 包：
# using Pkg
# Pkg.add(["HTTP", "JSON"])

# 加载 SymbolicRegression
# 如果项目已被 Pkg.activate() 激活，则直接使用
# 否则尝试从本地路径加载
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
    AbstractOptions,
    safe_pow
using SymbolicRegression.MutationFunctionsModule: mutate_constant, mutate_operator, mutate_feature
using DynamicExpressions: has_constants, parse_expression

using HTTP
using JSON

# ===========================================
# LLM API 配置
# ===========================================
const MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"  # 可选模型：
# - "meta-llama/Llama-3.1-8B-Instruct"
# - "mistralai/Mistral-7B-Instruct-v0.3"
# - "Qwen/Qwen3-32B"
# - "openai/gpt-oss-120b"

const API_BASE_URL = "https://ollama.cs.odu.edu"
const API_KEY = "sk-704f0fe8734549c9b57ac09a69001ad7"

"""
    call_llm(prompt::String; timeout::Int=120, max_retries::Int=3) -> String

调用LLM API，发送prompt并返回响应。

# 参数
- `timeout`: 超时时间（秒），默认120秒
- `max_retries`: 最大重试次数，默认3次
"""
function call_llm(prompt::String; timeout::Int=120, max_retries::Int=3)
    url = "$API_BASE_URL/api/v1/chat/completions"
    
    headers = Dict(
        "Content-Type" => "application/json",
        "Authorization" => "Bearer $API_KEY"
    )
    
    payload = Dict(
        "model" => MODEL_NAME,
        "messages" => [
            Dict("role" => "user", "content" => prompt)
        ],
        "stream" => false
    )
    
    last_error = nothing
    
    for attempt in 1:max_retries
        try
            response = HTTP.post(
                url,
                headers=headers,
                body=JSON.json(payload);
                readtimeout=timeout,
                connecttimeout=30  # 连接超时30秒
            )
            
            if response.status == 200
                body_str = String(response.body)
                result = JSON.parse(body_str)
                if haskey(result, "choices") && length(result["choices"]) > 0
                    message = result["choices"][1]["message"]["content"]
                    return message
                else
                    return "Error: No choices in response - $(body_str[1:min(200, length(body_str))])"
                end
            else
                error_msg = "Error: $(response.status) - $(String(response.body)[1:min(200, length(String(response.body)))])"
                if attempt < max_retries
                    @warn "API调用失败 (尝试 $(attempt)/$(max_retries)): $(error_msg)，正在重试..."
                    sleep(2)  # 等待2秒后重试
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
                sleep(2)  # 等待2秒后重试
            else
                if isa(e, HTTP.Exceptions.TimeoutError)
                    return "Request failed: TimeoutError after $(max_retries) attempts (timeout: $(timeout)s). The LLM may be taking too long to respond. Try increasing the timeout or simplifying the prompt."
                else
                    return "Request failed after $(max_retries) attempts: $(e)"
                end
            end
        end
    end
    
    return "Request failed: Unknown error"
end

"""
    format_equations_for_llm(dominating::Vector{PopMember}, options::AbstractOptions, dataset::Dataset)

将Pareto前沿的表达式格式化为LLM可读的格式。
"""
function format_equations_for_llm(dominating::Vector, options, dataset)
    lines = String[]
    push!(lines, "当前找到的最优表达式（按复杂度排序）：")
    push!(lines, "")
    push!(lines, "复杂度\t损失(Loss)\t分数(Score)\t表达式")
    push!(lines, "-" ^ 80)
    
    for member in dominating
        complexity = compute_complexity(member, options)
        loss = member.loss
        score = -member.loss  # score通常是负的loss
        equation = string_tree(member.tree, options)
        push!(lines, "$complexity\t$loss\t$score\t$equation")
    end
    
    return join(lines, "\n")
end

"""
    create_llm_prompt(equations_text::String, iteration::Int, max_iterations::Int, target_count::Int, population_size::Int) -> String

创建发送给LLM的prompt，要求LLM不仅选择表达式，还要生成新的表达式。

# 参数
- `target_count`: LLM应该选择的表达式数量
- `population_size`: 种群大小（需要生成的总数）
"""
function create_llm_prompt(equations_text::String, iteration::Int, max_iterations::Int, target_count::Int, population_size::Int)
    new_expressions_needed = population_size - target_count
    
    prompt = """
你是一个符号回归专家。当前正在进行符号回归搜索，这是第 $iteration/$max_iterations 轮。

以下是当前找到的最优表达式：

$equations_text

**任务1：分析表达式类别**
请先分析这些表达式属于什么类别（例如：多项式、指数函数、三角函数、有理函数等），并识别它们的共同模式和结构特征。

**任务2：选择现有表达式**
请根据以下标准选择 $target_count 个最有希望的表达式：
1. 损失(Loss)较低 - 表示拟合效果好
2. 复杂度适中 - 不要太简单（可能欠拟合）也不要太复杂（可能过拟合）
3. 表达式结构有潜力 - 可能通过变异和交叉产生更好的结果
4. 多样性 - 尽量选择不同结构的表达式

**任务3：生成新表达式**
基于你的分析，请生成 $new_expressions_needed 个新的表达式变体。这些新表达式应该：
1. 与选中的表达式属于相似的类别或模式
2. 具有相似的复杂度（不要过于简单或复杂）
3. 探索不同的结构变体（例如：改变运算符顺序、添加/移除项、调整常数等）
4. 使用相同的变量（x1, x2, x3等）和运算符（+, *, -, /, ^）

**重要提示：**
- 表达式必须使用变量名 x1, x2, x3 等（根据数据维度）
- 支持的运算符：+ (加法), - (减法), * (乘法), / (除法), ^ (幂运算)
- 可以使用常数（如 2.5, -1.3 等）
- 表达式应该简洁且有效

请以JSON格式返回，格式如下：
{
  "selected_indices": [1, 3, 5],
  "category_analysis": "这些表达式主要是多项式类型，包含 x2^2 项和线性项 x1...",
  "new_expressions": [
    "x1 + x2 * x2",
    "(x2 * x2) * 2.5 + x1",
    "x1 * x2 + x2 * x2"
  ],
  "reasoning": "选择这些表达式的原因，以及生成新表达式的思路..."
}

其中：
- selected_indices: 选中的表达式索引（从1开始），应该包含 $target_count 个索引
- category_analysis: 对表达式类别的分析
- new_expressions: 新生成的表达式字符串数组，应该包含 $new_expressions_needed 个表达式
- reasoning: 你的推理过程
"""
    return prompt
end

"""
    parse_llm_response(response::String) -> Tuple{Vector{Int}, Vector{String}}

解析LLM的响应，提取选择的表达式索引和新生成的表达式。

返回: (selected_indices, new_expressions)
"""
function parse_llm_response(response::String)
    try
        # 方法1: 直接使用正则表达式提取JSON中的数组（最可靠）
        pattern = r"\"selected_indices\"\s*:\s*\[([^\]]+)\]"
        match_result = match(pattern, response)
        if match_result !== nothing
            indices_str = match_result.captures[1]
            # 提取所有数字
            numbers = Int[]
            for m in eachmatch(r"\d+", indices_str)
                try
                    push!(numbers, parse(Int, m.match))
                catch
                    continue
                end
            end
            if !isempty(numbers)
                selected_indices = unique(numbers)
                new_expressions = String[]
                # 尝试提取新表达式
                new_expr_pattern = r"\"new_expressions\"\s*:\s*\[(.*?)\]"
                new_expr_match = match(new_expr_pattern, response, multiline=true)
                if new_expr_match !== nothing
                    expr_str = new_expr_match.captures[1]
                    # 提取引号内的字符串
                    for m in eachmatch(r"\"([^\"]+)\"", expr_str)
                        push!(new_expressions, m.captures[1])
                    end
                end
                return (selected_indices, new_expressions)
            end
        end
        
        # 方法2: 尝试JSON解析
        json_start = findfirst("{", response)
        json_end = findlast("}", response)
        
        if json_start !== nothing && json_end !== nothing
            json_str = response[json_start:json_end]
            result = JSON.parse(json_str)
            
            selected_indices = Int[]
            new_expressions = String[]
            
            if haskey(result, "selected_indices")
                indices = result["selected_indices"]
                if indices isa Vector && length(indices) > 0
                    # 安全地转换为Int数组
                    for idx in indices
                        try
                            idx_str = string(idx)
                            num_match = match(r"\d+", idx_str)
                            if num_match !== nothing
                                push!(selected_indices, parse(Int, num_match.match))
                            end
                        catch
                            continue
                        end
                    end
                end
            end
            
            if haskey(result, "new_expressions")
                exprs = result["new_expressions"]
                if exprs isa Vector
                    for expr in exprs
                        if expr isa String
                            push!(new_expressions, expr)
                        end
                    end
                end
            end
            
            if !isempty(selected_indices) || !isempty(new_expressions)
                return (unique(selected_indices), new_expressions)
            end
        end
        
        # 方法3: 最后尝试：从整个响应中提取所有数字（但只取合理的索引范围）
        numbers = Int[]
        for m in eachmatch(r"\b(\d+)\b", response)
            try
                num = parse(Int, m.match)
                # 只保留合理的索引范围（1-20）
                if 1 <= num <= 20
                    push!(numbers, num)
                end
            catch
                continue
            end
        end
        
        # 如果所有方法都失败，返回空数组
        return (isempty(numbers) ? Int[] : unique(numbers), String[])
    catch e
        @warn "无法解析LLM响应: $e"
        # 打印响应以便调试
        @warn "响应内容: $(response[1:min(500, length(response))])"
        return (Int[], String[])
    end
end

"""
    parse_llm_generated_expressions(expr_strings::Vector{String}, dataset::Dataset, options::AbstractOptions) -> Vector{PopMember}

解析LLM生成的新表达式字符串，转换为PopMember对象。
"""
function parse_llm_generated_expressions(
    expr_strings::Vector{String},
    dataset::Dataset,
    options::AbstractOptions
)
    members = PopMember[]
    
    for expr_str in expr_strings
        try
            # 解析表达式字符串
            parsed_expr = parse_expression(
                expr_str;
                operators=options.operators,
                variable_names=dataset.variable_names,
                node_type=options.node_type,
                expression_type=options.expression_type,
            )
            
            # 创建PopMember
            member = PopMember(
                dataset,
                parsed_expr,
                options;
                deterministic=options.deterministic,
            )
            push!(members, member)
        catch e
            @warn "无法解析LLM生成的表达式 '$expr_str': $e"
            continue
        end
    end
    
    return members
end

"""
    create_custom_population_from_selected(
        selected_members::Vector{PopMember},
        dataset::Dataset,
        options::AbstractOptions,
        population_size::Int,
        nfeatures::Int;
        use_only_selected::Bool=false,
        min_complexity::Union{Int,Nothing}=nothing
    ) -> Population

从选中的成员创建自定义种群。

# 参数
- `use_only_selected`: 如果为true，只使用选中的成员（通过复制填满种群）
- `min_complexity`: 如果指定，随机成员的最小复杂度（避免生成太简单的表达式）
- `min_complexity_for_selection`: 如果指定，只考虑复杂度 >= 此值的表达式进行LLM选择（过滤掉低复杂度表达式）
"""
function create_custom_population_from_selected(
    selected_members::Vector{<:PopMember},
    dataset::Dataset,
    options::AbstractOptions,
    population_size::Int,
    nfeatures::Int;
    use_only_selected::Bool=false,
    min_complexity::Union{Int,Nothing}=nothing
)
    # 先创建一个随机种群以确定正确的类型
    temp_pop = Population(
        dataset;
        population_size=1,
        nlength=3,
        options=options,
        nfeatures=nfeatures,
    )
    custom_members = typeof(temp_pop.members[1])[]
    
    # 添加选中的成员（重新创建以确保类型匹配）
    for member in selected_members
        new_member = PopMember(
            dataset,
            member.tree.tree,  # 使用原始树节点
            options;
            deterministic=options.deterministic,
        )
        push!(custom_members, new_member)
    end
    
    if use_only_selected
        # 只使用选中的成员，通过复制和轻微变异填满种群
        # 首先添加所有选中的成员
        # 然后对选中的成员进行轻微变异，生成变体
        while length(custom_members) < population_size
            # 随机选择一个已选中的成员
            idx = rand(1:length(selected_members))
            base_member = selected_members[idx]
            
            # 决定是直接复制还是进行轻微变异
            if length(custom_members) < length(selected_members) || rand() < 0.3
                # 直接复制（保留原始表达式）
                new_member = PopMember(
                    dataset,
                    base_member.tree.tree,
                    options;
                    deterministic=options.deterministic,
                )
            else
                # 进行轻微变异（30%概率进行变异，70%概率直接复制）
                try
                    # 复制树
                    mutated_tree = copy(base_member.tree.tree)
                    
                    # 随机选择一种轻微的变异操作
                    mutation_type = rand([:constant, :operator, :feature])
                    if mutation_type == :constant && has_constants(mutated_tree)
                        # 小幅度变异常数（temperature=0.1表示小幅度变异）
                        mutated_tree = mutate_constant(mutated_tree, 0.1, options)
                    elseif mutation_type == :operator
                        mutated_tree = mutate_operator(mutated_tree, options)
                    elseif mutation_type == :feature
                        mutated_tree = mutate_feature(mutated_tree, nfeatures)
                    end
                    
                    # 创建新的PopMember
                    new_member = PopMember(
                        dataset,
                        mutated_tree,
                        options;
                        deterministic=options.deterministic,
                    )
                catch
                    # 如果变异失败，直接复制
                    new_member = PopMember(
                        dataset,
                        base_member.tree.tree,
                        options;
                        deterministic=options.deterministic,
                    )
                end
            end
            
            push!(custom_members, new_member)
        end
    else
        # 补充随机成员
        if length(custom_members) < population_size
            # 计算选中成员的平均复杂度，用于生成相似复杂度的随机成员
            avg_complexity = if !isempty(selected_members)
                sum(compute_complexity(m, options) for m in selected_members) / length(selected_members)
            else
                5.0
            end
            
            # 根据平均复杂度设置nlength（树长度）
            # 复杂度大致与树长度相关
            target_nlength = max(3, Int(round(avg_complexity / 2)))
            if min_complexity !== nothing
                target_nlength = max(target_nlength, Int(round(min_complexity / 2)))
            end
            
            random_pop = Population(
                dataset;
                population_size=population_size - length(custom_members),
                nlength=target_nlength,
                options=options,
                nfeatures=nfeatures,
            )
            append!(custom_members, random_pop.members)
        end
    end
    
    return Population(custom_members)
end

"""
    symbolic_regression_with_llm(
        X::AbstractMatrix,
        y::AbstractVector,
        options::AbstractOptions;
        max_iterations::Int=5,
        iterations_per_round::Int=10,
        llm_selection_count::Int=5,
        verbosity::Int=1
    )

使用LLM辅助的符号回归搜索。

# 参数
- `X`: 输入数据矩阵
- `y`: 目标值向量
- `options`: SymbolicRegression选项
- `max_iterations`: 最大循环轮数（每次循环包括搜索+LLM选择）
- `iterations_per_round`: 每轮搜索的迭代次数
- `llm_selection_count`: LLM每次选择的表达式数量
- `verbosity`: 详细程度（0=静默，1=正常，2=详细）
- `use_only_selected`: 如果为true，只使用LLM选中的表达式（通过复制填满种群），不添加随机成员
- `min_complexity`: 如果指定，随机成员的最小复杂度（避免生成太简单的表达式）
- `min_complexity_for_selection`: 如果指定，只考虑复杂度 >= 此值的表达式进行LLM选择（过滤掉低复杂度表达式）
"""
function symbolic_regression_with_llm(
    X::AbstractMatrix{T},
    y::AbstractVector{T},
    options::AbstractOptions;
    max_iterations::Int=5,
    iterations_per_round::Int=10,
    llm_selection_count::Int=5,
    verbosity::Int=1,
    use_only_selected::Bool=false,
    min_complexity::Union{Int,Nothing}=nothing,
    min_complexity_for_selection::Union{Int,Nothing}=nothing,
) where {T}
    
    # 确保加载SymbolicRegression模块
    # 假设已经在外部加载了
    # using SymbolicRegression
    
    # 创建数据集
    dataset = Dataset(X, y)
    nfeatures = size(X, 1)
    
    verbosity > 0 && println("=" ^ 80)
    verbosity > 0 && println("开始LLM辅助的符号回归搜索")
    verbosity > 0 && println("=" ^ 80)
    verbosity > 0 && println("配置:")
    verbosity > 0 && println("  - 最大循环轮数: $max_iterations")
    verbosity > 0 && println("  - 每轮迭代次数: $iterations_per_round")
    verbosity > 0 && println("  - LLM选择数量: $llm_selection_count")
    verbosity > 0 && println("  - 种群大小: $(options.population_size)")
    verbosity > 0 && println()
    
    current_population = nothing
    best_hof = nothing
    
    for iteration in 1:max_iterations
        verbosity > 0 && println("=" ^ 80)
        verbosity > 0 && println("第 $iteration/$max_iterations 轮")
        verbosity > 0 && println("=" ^ 80)
        
        # 步骤1: 进行搜索
        verbosity > 0 && println("\n[步骤1] 进行符号回归搜索...")
        hof = if current_population === nothing
            # 第一轮：使用随机初始种群
            equation_search(
                X, y;
                options=options,
                niterations=iterations_per_round,
                parallelism=:serial,
                verbosity=verbosity > 1 ? verbosity : 0,
            )
        else
            # 后续轮：使用LLM选择的自定义种群
            equation_search(
                X, y;
                options=options,
                initial_population=current_population,
                niterations=iterations_per_round,
                parallelism=:serial,
                verbosity=verbosity > 1 ? verbosity : 0,
            )
        end
        
        # 更新最佳结果
        if best_hof === nothing
            best_hof = hof
        else
            # 合并结果，保留最好的
            for size in 1:options.maxsize
                if hof.exists[size] && (!best_hof.exists[size] || hof.members[size].cost < best_hof.members[size].cost)
                    best_hof.members[size] = hof.members[size]
                    best_hof.exists[size] = true
                end
            end
        end
        
        # 计算Pareto前沿
        dominating = calculate_pareto_frontier(hof)
        
        # 如果指定了最小复杂度，过滤掉低复杂度的表达式
        if min_complexity_for_selection !== nothing
            dominating = [
                member for member in dominating 
                if compute_complexity(member, options) >= min_complexity_for_selection
            ]
        end
        
        verbosity > 0 && println("\n找到 $(length(dominating)) 个Pareto最优表达式" * 
            (min_complexity_for_selection !== nothing ? " (复杂度 >= $min_complexity_for_selection)" : ""))
        
        # 显示当前最佳结果
        if verbosity > 0
            println("\n当前最佳表达式:")
            for member in dominating
                complexity = compute_complexity(member, options)
                loss = member.loss
                equation = string_tree(member.tree, options)
                println("  复杂度=$complexity, 损失=$loss, 表达式=$equation")
            end
        end
        
        # 如果没有符合条件的表达式，跳过LLM选择
        if isempty(dominating)
            verbosity > 0 && @warn "没有找到复杂度 >= $(min_complexity_for_selection) 的表达式，跳过LLM选择"
            if iteration == max_iterations
                verbosity > 0 && println("\n达到最大迭代次数，搜索结束。")
                break
            end
            # 继续下一轮，但使用随机种群
            current_population = nothing
            continue
        end
        
        # 如果是最后一轮，不需要LLM选择
        if iteration == max_iterations
            verbosity > 0 && println("\n达到最大迭代次数，搜索结束。")
            break
        end
        
        # 步骤2: 格式化结果并发送给LLM
        verbosity > 0 && println("\n[步骤2] 将结果发送给LLM进行选择...")
        equations_text = format_equations_for_llm(dominating, options, dataset)
        
        # 动态计算LLM应该选择的表达式数量
        # 如果use_only_selected=true，让LLM选择更多表达式（种群大小的20-50%）
        # 否则使用llm_selection_count
        target_selection_count = if use_only_selected
            # 选择种群大小的20-50%，但不超过可用表达式数量
            min(
                max(llm_selection_count, Int(round(options.population_size * 0.2))),
                min(Int(round(options.population_size * 0.5)), length(dominating))
            )
        else
            llm_selection_count
        end
        
        prompt = create_llm_prompt(equations_text, iteration, max_iterations, target_selection_count, options.population_size)
        
        if verbosity > 1
            println("\n发送给LLM的Prompt:")
            println("-" ^ 80)
            println(prompt)
            println("-" ^ 80)
        end
        
        llm_response = call_llm(prompt)
        verbosity > 0 && println("LLM响应:")
        verbosity > 0 && println(llm_response)
        verbosity > 0 && println()
        
        # 步骤3: 解析LLM响应并选择表达式
        verbosity > 0 && println("[步骤3] 解析LLM响应并选择表达式...")
        selected_indices, new_expressions = parse_llm_response(llm_response)
        
        if isempty(selected_indices)
            verbosity > 0 && @warn "LLM未选择任何表达式，使用前 $llm_selection_count 个最佳表达式"
            selected_indices = collect(1:min(llm_selection_count, length(dominating)))
        else
            # 确保索引在有效范围内
            selected_indices = [i for i in selected_indices if 1 <= i <= length(dominating)]
            if isempty(selected_indices)
                verbosity > 0 && @warn "LLM选择的索引无效，使用前 $llm_selection_count 个最佳表达式"
                selected_indices = collect(1:min(llm_selection_count, length(dominating)))
            end
        end
        
        verbosity > 0 && println("LLM选择了索引: $selected_indices (共 $(length(selected_indices)) 个表达式)")
        if !isempty(new_expressions)
            verbosity > 0 && println("LLM生成了 $(length(new_expressions)) 个新表达式")
        end
        
        # 步骤4: 创建自定义种群
        verbosity > 0 && println("\n[步骤4] 创建自定义种群...")
        selected_members = [dominating[i] for i in selected_indices]
        
        # 解析LLM生成的新表达式
        llm_generated_members = if !isempty(new_expressions)
            parse_llm_generated_expressions(new_expressions, dataset, options)
        else
            PopMember[]
        end
        
        if !isempty(llm_generated_members)
            verbosity > 0 && println("成功解析 $(length(llm_generated_members)) 个LLM生成的表达式")
            # 将LLM生成的成员添加到选中的成员中
            append!(selected_members, llm_generated_members)
        end
        
        # 如果LLM生成的表达式已经足够填满种群，直接使用
        if length(selected_members) >= options.population_size
            # 只取前population_size个
            selected_members = selected_members[1:options.population_size]
            current_population = Population(selected_members)
            verbosity > 0 && println("成功创建自定义种群，包含 $(length(selected_members)) 个表达式（全部来自LLM选择和生成）")
        else
            # 需要补充或变异来填满种群
            current_population = create_custom_population_from_selected(
                selected_members,
                dataset,
                options,
                options.population_size,
                nfeatures;
                use_only_selected=use_only_selected,
                min_complexity=min_complexity,
            )
            
            if use_only_selected
                verbosity > 0 && println("成功创建自定义种群，包含 $(length(selected_indices)) 个LLM选择的表达式和 $(length(llm_generated_members)) 个LLM生成的表达式（通过复制/变异填满到 $(options.population_size) 个成员）")
            else
                verbosity > 0 && println("成功创建自定义种群，包含 $(length(selected_indices)) 个LLM选择的表达式、$(length(llm_generated_members)) 个LLM生成的表达式和 $(options.population_size - length(selected_members)) 个随机成员")
            end
        end
        verbosity > 0 && println()
    end
    
    verbosity > 0 && println("=" ^ 80)
    verbosity > 0 && println("搜索完成！")
    verbosity > 0 && println("=" ^ 80)
    
    # 返回最终的最佳结果
    final_dominating = calculate_pareto_frontier(best_hof)
    
    # 如果指定了最小复杂度，过滤掉低复杂度的表达式
    if min_complexity_for_selection !== nothing
        final_dominating = [
            member for member in final_dominating 
            if compute_complexity(member, options) >= min_complexity_for_selection
        ]
    end
    
    verbosity > 0 && println("\n最终找到的最优表达式" * 
        (min_complexity_for_selection !== nothing ? " (复杂度 >= $min_complexity_for_selection)" : "") * ":")
    for member in final_dominating
        complexity = compute_complexity(member, options)
        loss = member.loss
        equation = string_tree(member.tree, options)
        verbosity > 0 && println("  复杂度=$complexity, 损失=$loss, 表达式=$equation")
    end
    
    return best_hof
end

# ===========================================
# 示例使用
# ===========================================
if abspath(PROGRAM_FILE) == @__FILE__
    println("=" ^ 80)
    println("SymbolicRegression + LLM 集成示例")
    println("=" ^ 80)
    println()
    
    # 1. 准备数据
    println("准备数据...")
    X = randn(Float32, 3, 100)
    y = Float32(2.0) * X[1, :] .+ Float32(3.0) * X[2, :] .^ 2 .- Float32(1.5)
    println("数据形状: X=$(size(X)), y=$(size(y))")
    println()
    
    # 2. 设置选项
    println("设置SymbolicRegression选项...")
    options = Options(;
        binary_operators=[+, *, -, /, safe_pow],
        unary_operators=[],
        population_size=50,
        maxsize=10,
        verbosity=0,  # 在循环中控制详细程度
    )
    println("选项设置完成")
    println()
    
    # 3. 运行LLM辅助的搜索
    println("开始LLM辅助的符号回归搜索...")
    println()
    
    final_hof = symbolic_regression_with_llm(
        X, y,
        options;
        max_iterations=3,  # 总共3轮循环
        iterations_per_round=10,  # 每轮搜索10次迭代
        llm_selection_count=5,  # LLM每次选择5个表达式
        verbosity=1,  # 显示进度
    )
    
    println()
    println("=" ^ 80)
    println("完成！")
    println("=" ^ 80)
end

