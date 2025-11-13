# 1. 设置开发环境并安装依赖
using Pkg

# 激活 SymbolicRegression 项目（确保所有依赖都被安装）
sr_project_path = joinpath(@__DIR__, "..")
Pkg.activate(sr_project_path)
Pkg.instantiate()  # 安装所有在 Project.toml 中声明的依赖

# 安装额外的依赖（HTTP 和 JSON）
Pkg.add(["HTTP", "JSON"])

# 2. 运行示例
include("llm_integration.jl")

# 3. 准备数据和选项
X = randn(Float32, 3, 100)
y = Float32(2.0) * X[1, :] .+ Float32(3.0) * X[2, :] .^ 2 .- Float32(1.5)

options = Options(;
    binary_operators=[+, *, -, /, safe_pow],
    population_size=50,
    maxsize=10,
)

# 4. 运行LLM辅助的搜索
final_hof = symbolic_regression_with_llm(
    X, y, options;
    max_iterations=5,        # 总共5轮循环
    iterations_per_round=40, # 每轮搜索10次迭代
    llm_selection_count=3,   # LLM每次选择5个表达式
    verbosity=1,
    use_only_selected=true,  # 如果为true，只使用选中的表达式（不添加随机成员）
    min_complexity=3,         # 随机成员的最小复杂度（避免生成太简单的表达式）
    min_complexity_for_selection=3,  # 只考虑复杂度 >= 3 的表达式进行LLM选择
)