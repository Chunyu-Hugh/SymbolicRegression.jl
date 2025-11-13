"""
示例：如何使用自定义初始种群

这个示例展示了如何创建自定义初始种群，并将其传递给equation_search函数。

⚠️ 重要提示：
如果你修改了 SymbolicRegression.jl 的源代码，你需要让 Julia 使用本地开发版本。
在运行此脚本之前，请在 Julia REPL 中执行：

    using Pkg
    Pkg.develop(path="C:/Users/cs_chu034/download/projects/SR_LLM/SymbolicRegression.jl")
    # 或者使用相对路径（从项目根目录）
    # Pkg.develop(path="./SymbolicRegression.jl")

然后重新加载模块：
    using SymbolicRegression

查看 examples/README_DEVELOPMENT.md 获取更多信息。
"""

# 确保使用本地开发版本
# 如果遇到依赖冲突（如 LibraryAugmentedSymbolicRegression），可以使用以下方法：

# 方法1: 移除冲突的包（推荐）
# using Pkg
# Pkg.rm("LibraryAugmentedSymbolicRegression")
# Pkg.develop(path=joinpath(@__DIR__, ".."))
# using SymbolicRegression

# 方法2: 创建新环境（推荐用于开发）
# using Pkg
# Pkg.activate(".")
# Pkg.develop(path=joinpath(@__DIR__, ".."))
# Pkg.instantiate()
# using SymbolicRegression

# 方法3: 直接使用（如果已解决依赖冲突）
# 注意：如果遇到 "no method matching" 错误，说明仍在使用已安装的包版本
# 需要先执行方法1或方法2，或者重启 Julia 并重新加载

using SymbolicRegression
using DynamicExpressions: parse_expression

# 验证是否使用了本地版本（可选）
try
    sr_path = pathof(SymbolicRegression)
    local_path = joinpath(@__DIR__, "..", "src", "SymbolicRegression.jl")
    if occursin(abspath(local_path), sr_path)
        println("✓ 正在使用本地开发版本: ", sr_path)
    else
        @warn "⚠ 可能仍在使用已安装的包版本: ", sr_path
        @warn "   如果遇到 'no method matching' 错误，请使用方法1或方法2加载本地版本"
    end
catch
    # 忽略错误
end

# 1. 准备数据
# 数据集包含 5 个特征，目标由以下公式生成：
# y = 2 * cos(x4) + x1^2 - 2 + 噪声
X = randn(Float64, 5, 100)
y = 2 .* cos.(X[4, :]) .+ X[1, :].^2 .- 2
y = y .+ randn(100) .* 1e-3

# 2. 设置选项
# 注意：需要使用 safe_pow 而不是 ^，因为 SymbolicRegression 使用 safe_pow 来处理幂运算
using SymbolicRegression: safe_pow
options = Options(;
    binary_operators=[+, *, -, /, safe_pow],  # 使用 safe_pow 而不是 ^
    unary_operators=[cos],
    population_size=20,  # 种群大小
    maxsize=10,
)

# 3. 创建数据集
dataset = Dataset(X, y)

# 方法1: 从字符串表达式创建自定义种群
# println("=" ^ 80)
# println("方法1: 从字符串表达式创建自定义种群")
# println("=" ^ 80)
#
# # 创建一些自定义表达式
# # 注意：parse_expression 可能无法直接解析字符串中的 ^，所以这里使用简单的表达式
# # 如果需要幂运算，可以在创建 PopMember 后手动修改树结构，或者使用其他方式
custom_expressions = [
    "x1 + x2",
    "x1 + 2.0 * cos(x4)",
    "x1 - x3 + x5",
    "safe_pow(x1, 2) + x2",
    "x1 * x2 + x3",
    "x1 * x4 + x5",
    "x1 * x2 * x3",
    "(x1 + x2) * x4",
    "(x1 + x2 + x3) / (x4 + 1.0)",
    "x1 + x2 + x3 + x4 + x5",
    "(x1 - x2) * (x3 - x4)",
    "x1 / (1.0 + safe_pow(x2, 2)) + x5",
    "x1 + safe_pow(x3, 2) - 2.0",
    "2.0 * cos(x4) + x1",
    "2.0 * cos(x4) + safe_pow(x1, 2)",
    "(x1 + 0.5 * x5) * cos(x4)",
    "(x1 - x2) / (x5 + 1.0)",
    "(x1 + x2) * (x3 + x4)",
    "x1 + x2 * x5 + cos(x4)",
    "safe_pow(x1 + x2, 2) - x3 + cos(x4)",
]

#=
println("=" ^ 80)
println("方法1: 从字符串表达式创建自定义种群")
println("=" ^ 80)

# 将字符串表达式转换为PopMember对象
# 使用 guesses 参数来创建正确类型的成员（推荐方法）
println("使用 guesses 参数创建自定义成员...")
temp_hof = equation_search(
    X, y;
    options=options,
    guesses=custom_expressions,
    niterations=0,  # 不进行搜索，只创建 guesses
    parallelism=:serial,
)

# 从 HallOfFame 中提取自定义成员
# 注意：guesses 会被解析并添加到 HallOfFame 中，但可能不在前几个位置
# 我们需要找到所有存在的成员，并提取复杂度匹配的
custom_members_from_guesses = typeof(temp_hof.members[1])[]
for size in 1:options.maxsize
    if temp_hof.exists[size]
        # 检查这个成员是否来自我们的 guesses（通过检查复杂度是否合理）
        # 实际上，guesses 会被添加到 seed_members 并通过 migration 添加
        # 这里我们简单地提取所有存在的成员
        push!(custom_members_from_guesses, temp_hof.members[size])
        if size <= length(custom_expressions)
            println("✓ 从 guesses 创建成员 (复杂度=$size): $(custom_expressions[size])")
        end
    end
end

# 只保留前几个（匹配 guesses 数量）
if length(custom_members_from_guesses) > length(custom_expressions)
    custom_members_from_guesses = custom_members_from_guesses[1:min(length(custom_expressions), length(custom_members_from_guesses))]
end

# 创建完整的种群：自定义成员 + 随机成员
if length(custom_members_from_guesses) > 0
    # 先创建一个完整的随机种群
    random_pop = Population(
        dataset;
        population_size=options.population_size,
        nlength=3,
        options=options,
        nfeatures=size(X, 1),
    )
    # 使用随机种群的类型作为基准
    custom_members = typeof(random_pop.members[1])[]
    
    # 添加自定义成员（重新创建以确保类型匹配）
    for member in custom_members_from_guesses
        try
            # 尝试直接添加
            push!(custom_members, member)
        catch
            # 如果类型不匹配，重新创建
            new_member = PopMember(
                dataset,
                member.tree.tree,  # 使用原始树节点
                options;
                deterministic=options.deterministic,
            )
            push!(custom_members, new_member)
        end
    end
    
    # 添加随机成员
    for i in 1:(options.population_size - length(custom_members))
        push!(custom_members, random_pop.members[i])
    end
    
    println("成功创建 $(length(custom_members_from_guesses)) 个自定义成员，补充 $(options.population_size - length(custom_members_from_guesses)) 个随机成员")
else
    println("警告: 无法从 guesses 创建任何成员，使用完全随机种群")
    custom_members = Population(
        dataset;
        population_size=options.population_size,
        nlength=3,
        options=options,
        nfeatures=size(X, 1),
    ).members
end

# 创建自定义种群
custom_population = Population(custom_members)
println("\n自定义种群大小: $(length(custom_population.members))")

# 使用自定义种群进行搜索
println("\n开始使用自定义种群进行搜索...")
hall_of_fame = equation_search(
    X, y;
    options=options,
    initial_population=custom_population,
    parallelism=:serial,
    niterations=20,
)

# 显示结果
dominating = calculate_pareto_frontier(hall_of_fame)
println("\n找到的最优表达式:")
for member in dominating
    complexity = compute_complexity(member, options)
    loss = member.loss
    string = string_tree(member.tree, options)
    println("复杂度: $complexity, 损失: $loss, 表达式: $string")
end
=#

#=
println("\n" * "=" ^ 80)
println("方法2: 从已有的搜索结果创建自定义种群")
println("=" ^ 80)

# 先进行一次搜索
println("进行第一次搜索...")
hof1 = equation_search(X, y; options=options, niterations=10, parallelism=:serial)

# 从第一次搜索的结果中提取一些成员
# 先创建一个随机种群以确定正确的类型
temp_pop2 = Population(
    dataset;
    population_size=1,
    nlength=3,
    options=options,
    nfeatures=size(X, 1),
)
best_members = typeof(temp_pop2.members[1])[]

# 提取 HallOfFame 中的成员
# 注意：equation_search 返回的 HallOfFame 已经通过 embed_metadata 处理
# 但类型可能与 Population 创建的成员不同，所以我们需要重新创建成员以确保类型匹配
for size in 1:min(5, options.maxsize)
    if hof1.exists[size]
        member = hof1.members[size]
        # 重新创建 PopMember 以确保类型匹配
        # 使用现有成员的树节点，通过 create_expression 重新创建以确保类型正确
        new_member = PopMember(
            dataset,
            member.tree.tree,  # 使用原始树节点
            options;
            deterministic=options.deterministic,
        )
        push!(best_members, new_member)
        println("✓ 提取成员 (复杂度=$size): $(string_tree(new_member.tree, options))")
    end
end

# 如果没有提取到任何成员，创建一个随机种群
if isempty(best_members)
    println("警告: 无法从HallOfFame提取成员，使用随机种群")
    best_members = Population(
        dataset;
        population_size=options.population_size,
        nlength=3,
        options=options,
        nfeatures=size(X, 1),
    ).members
end

# 补充随机成员
if length(best_members) < options.population_size
    random_pop2 = Population(
        dataset;
        population_size=options.population_size - length(best_members),
        nlength=3,
        options=options,
        nfeatures=size(X, 1),
    )
    # 使用 vcat 而不是 append! 来避免类型转换问题
    best_members = vcat(best_members, random_pop2.members)
end

# 创建新的自定义种群
custom_population2 = Population(best_members)
println("\n使用改进的种群进行第二次搜索...")
hof2 = equation_search(
    X, y;
    options=options,
    initial_population=custom_population2,
    parallelism=:serial,
    niterations=20,
)

# 显示结果
dominating2 = calculate_pareto_frontier(hof2)
println("\n第二次搜索找到的最优表达式:")
for member in dominating2
    complexity = compute_complexity(member, options)
    loss = member.loss
    string = string_tree(member.tree, options)
    println("复杂度: $complexity, 损失: $loss, 表达式: $string")
end
=#

# 方法3: 手动创建PopMember对象
println("\n" * "=" ^ 80)
println("方法3: 手动创建PopMember对象")
println("=" ^ 80)

# 使用gen_random_tree创建一些随机树
using SymbolicRegression: gen_random_tree

# 先创建一个临时种群以确定正确的类型
temp_pop3 = Population(
    dataset;
    population_size=1,
    nlength=3,
    options=options,
    nfeatures=size(X, 1),
)
manual_members = typeof(temp_pop3.members[1])[]

# 检查表达式中是否包含特征变量（避免生成纯常数表达式）
contains_feature(tree) = any(tree) do node
    node.degree == 0 && !node.constant
end

# 生成带有变量的成员
while length(manual_members) < options.population_size
    tree = gen_random_tree(3, options, size(X, 1), Float64)
    contains_feature(tree) || continue
    member = PopMember(dataset, tree, options; deterministic=options.deterministic)
    push!(manual_members, member)
end

custom_population3 = Population(manual_members)
println("手动创建的种群大小: $(length(custom_population3.members))")

# 验证：打印初始种群中的前几个成员，用于后续对比
println("\n初始种群中的前5个成员（用于验证是否被使用）:")
initial_member_strings = String[]
for (idx, member) in enumerate(custom_population3.members[1:min(5, length(custom_population3.members))])
    member_str = string_tree(member.tree, options)
    complexity = compute_complexity(member, options)
    println("  [$idx] 复杂度=$complexity, 表达式: $member_str")
    push!(initial_member_strings, member_str)
end

# 验证种群大小是否匹配
if length(custom_population3.members) != options.population_size
    @warn "⚠️  警告: 自定义种群大小 ($(length(custom_population3.members))) 与 options.population_size ($(options.population_size)) 不匹配！"
    @warn "   系统会回退到随机种群，自定义种群不会被使用！"
else
    println("✓ 种群大小匹配 ($(length(custom_population3.members)) == $(options.population_size))，自定义种群应该会被使用")
end

println("\n开始使用手动种群进行搜索...")
hof3 = equation_search(
    X, y;
    options=options,
    initial_population=custom_population3,
    parallelism=:serial,
    niterations=20,
)

dominating3 = calculate_pareto_frontier(hof3)
println("\n第三次搜索找到的最优表达式:")
for member in dominating3
    complexity = compute_complexity(member, options)
    loss = member.loss
    string = string_tree(member.tree, options)
    println("复杂度: $complexity, 损失: $loss, 表达式: $string")
end

# 验证说明
println("\n" * "=" ^ 80)
println("验证说明:")
println("=" ^ 80)
println("""
根据 SymbolicRegression.jl 的源代码分析：

1. 自定义种群的使用流程：
   - equation_search 函数在第822-873行检查 initial_population 参数
   - 如果种群大小匹配，会使用 strip_metadata 处理自定义种群
   - 为每个成员重新计算损失（在新数据集上）
   - 通过 @sr_spawner 将自定义种群传递给进化循环
   - 在 _dispatch_s_r_cycle 中，自定义种群作为 in_pop 传入 s_r_cycle 进行进化

2. 验证方法：
   - 检查种群大小是否匹配（已在上方打印）
   - 如果大小不匹配，会看到警告信息，系统会回退到随机种群
   - 如果大小匹配，自定义种群会被使用，但成员会在进化过程中被修改/替换

3. 注意事项：
   - 即使自定义种群被使用，进化过程会通过变异、交叉等操作改变种群
   - 初始成员可能很快被更好的个体替换
   - 要确认是否真的使用了自定义种群，可以：
     a) 检查是否有警告信息（如果有警告，说明没使用）
     b) 设置 verbosity > 0 查看详细日志
     c) 使用 niterations=0 来只初始化种群而不进化，然后检查种群内容

4. 当前状态：
   - 种群大小: $(length(custom_population3.members))
   - options.population_size: $(options.population_size)
   - 匹配状态: $(length(custom_population3.members) == options.population_size ? "✓ 匹配" : "✗ 不匹配")
""")

println("\n" * "=" ^ 80)
println("方法3扩展: 使用高复杂度成员继续搜索")
println("=" ^ 80)

complexity_threshold = 7
selected_members = typeof(custom_population3.members[1])[]
for size in reverse(1:options.maxsize)
    if hof3.exists[size] && size >= complexity_threshold
        member = hof3.members[size]
        new_member = PopMember(
            dataset,
            member.tree.tree,
            options;
            deterministic=options.deterministic,
        )
        push!(selected_members, new_member)
        println("✓ 选取成员 (复杂度=$size): $(string_tree(new_member.tree, options))")
    end
end

if isempty(selected_members)
    println("警告: 未找到满足复杂度阈值的成员，改用方法3结果中的前几个成员")
    for size in reverse(1:options.maxsize)
        if hof3.exists[size]
            member = hof3.members[size]
            new_member = PopMember(
                dataset,
                member.tree.tree,
                options;
                deterministic=options.deterministic,
            )
            push!(selected_members, new_member)
        end
        length(selected_members) >= min(options.population_size, 5) && break
    end
end

while length(selected_members) < options.population_size
    idx = (length(selected_members) % length(manual_members)) + 1
    backup_member = manual_members[idx]
    push!(
        selected_members,
        PopMember(
            dataset,
            backup_member.tree.tree,
            options;
            deterministic=options.deterministic,
        ),
    )
end

custom_population4 = Population(selected_members[1:options.population_size])
println("高复杂度种群大小: $(length(custom_population4.members))")

println("使用高复杂度成员再次进行搜索...")
hof4 = equation_search(
    X, y;
    options=options,
    initial_population=custom_population4,
    parallelism=:serial,
    niterations=20,
)

dominating4 = calculate_pareto_frontier(hof4)
println("\n再次搜索的最优表达式:")
for member in dominating4
    complexity = compute_complexity(member, options)
    loss = member.loss
    string = string_tree(member.tree, options)
    println("复杂度: $complexity, 损失: $loss, 表达式: $string")
end

println("\n" * "=" ^ 80)
println("总结:")
println("=" ^ 80)
println("""
你可以通过以下方式创建自定义初始种群：

1. 从字符串表达式创建：使用parse_expression解析字符串，然后创建PopMember
2. 从已有搜索结果创建：从HallOfFame中提取成员
3. 手动创建：使用gen_random_tree或其他方法创建表达式树

注意事项：
- 种群大小必须与options.population_size匹配
- 如果种群大小不匹配，系统会回退到随机种群
- 自定义种群中的成员会在新的数据集上重新评估损失
- 可以结合guesses参数使用，guesses会通过migration机制添加到种群中
""")

