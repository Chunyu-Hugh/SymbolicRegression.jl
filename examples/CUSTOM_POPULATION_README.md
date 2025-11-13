# 自定义初始种群使用指南

## 概述

现在SymbolicRegression支持自定义初始种群！你可以直接指定初始种群，而不是使用随机生成的种群。

## 如何调用SymbolicRegression.jl

是的，**SymbolicRegression.jl是Julia代码，需要通过Julia来调用**。有以下几种方式：

### 方式1: 直接在Julia中使用

```julia
using SymbolicRegression

X = randn(Float32, 3, 100)
y = 2.0 * X[1, :] .+ 3.0 * X[2, :] .^ 2

options = Options(; population_size=50, maxsize=10)
hall_of_fame = equation_search(X, y; options=options, niterations=20)
```

### 方式2: 通过PySR（Python接口）

如果你使用Python，可以通过PySR来调用：

```python
from pysr import PySRRegressor

model = PySRRegressor(
    population_size=50,
    maxsize=10,
    niterations=20,
)
model.fit(X, y)
```

**注意**: PySR是Python包装器，底层调用Julia代码。目前PySR可能还不支持`initial_population`参数，你需要直接使用Julia代码来使用这个功能。

### 方式3: 在Julia脚本中运行

创建一个`.jl`文件，然后运行：

```bash
julia your_script.jl
```

## 使用自定义初始种群

### 基本用法

```julia
using SymbolicRegression
using DynamicExpressions: parse_expression

# 准备数据
X = randn(Float32, 3, 100)
y = 2.0 * X[1, :] .+ 3.0 * X[2, :] .^ 2

# 设置选项
options = Options(;
    binary_operators=[+, *, -, /],
    population_size=50,
    maxsize=10,
)

dataset = Dataset(X, y)

# 创建自定义种群
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

# 补充随机成员以填满种群
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

# 使用自定义种群
hall_of_fame = equation_search(
    X, y;
    options=options,
    initial_population=custom_population,
    niterations=20,
)
```

### 参数说明

`initial_population`参数可以是：

1. **单个Population对象**: 用于所有输出和所有种群
   ```julia
   initial_population=my_population
   ```

2. **Population向量**: 
   - 如果长度等于输出数量，每个输出使用对应的种群
   - 如果长度等于种群数量，每个种群使用对应的种群
   - 否则，所有都使用第一个种群

3. **nothing** (默认): 使用随机生成的种群

### 重要注意事项

1. **种群大小必须匹配**: 自定义种群的大小必须等于`options.population_size`，否则系统会回退到随机种群并发出警告。

2. **损失会重新计算**: 即使自定义种群中的成员已经有损失值，它们也会在新的数据集上重新评估。

3. **与guesses的区别**:
   - `guesses`: 通过migration机制添加到种群中（不会替换整个种群）
   - `initial_population`: 直接替换整个初始种群

4. **可以结合使用**: 你可以同时使用`guesses`和`initial_population`，guesses会通过migration机制在搜索过程中添加到种群中。

## 完整示例

查看 `examples/custom_initial_population.jl` 获取完整示例，包括：

1. 从字符串表达式创建自定义种群
2. 从已有搜索结果创建自定义种群
3. 手动创建PopMember对象

运行示例：

```bash
julia examples/custom_initial_population.jl
```

## 常见问题

### Q: 我可以在Python中使用这个功能吗？

A: 目前PySR（Python接口）可能还不支持`initial_population`参数。你需要直接使用Julia代码，或者等待PySR更新。

### Q: 如何从Python传递自定义种群到Julia？

A: 这是一个高级用法，需要：
1. 在Python中创建表达式字符串
2. 通过某种方式（如JSON）传递给Julia
3. 在Julia中解析并创建Population对象

或者，你可以直接在Julia中完成整个流程。

### Q: 自定义种群的大小可以小于population_size吗？

A: 不可以。如果大小不匹配，系统会回退到随机种群。你需要补充随机成员来填满种群。

### Q: 如何从之前的搜索结果创建自定义种群？

A: 从HallOfFame中提取成员：

```julia
# 第一次搜索
hof1 = equation_search(X, y; options=options, niterations=10)

# 提取最佳成员
best_members = []
for size in 1:min(5, options.maxsize)
    if hof1.exists[size]
        push!(best_members, hof1.members[size])
    end
end

# 补充随机成员
# ... (见上面的示例)

# 创建新种群并继续搜索
custom_pop = Population(best_members)
hof2 = equation_search(X, y; options=options, initial_population=custom_pop, niterations=20)
```

## 总结

- ✅ SymbolicRegression.jl是Julia代码，需要通过Julia调用
- ✅ 现在支持通过`initial_population`参数自定义初始种群
- ✅ 可以结合`guesses`参数使用
- ⚠️ 种群大小必须匹配`options.population_size`
- 📝 查看`examples/custom_initial_population.jl`获取完整示例

