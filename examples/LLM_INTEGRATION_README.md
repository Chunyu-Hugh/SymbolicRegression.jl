# SymbolicRegression + LLM 集成使用指南

## 概述

这个示例展示了如何将大语言模型（LLM）与SymbolicRegression结合，实现智能的符号回归搜索。

## 工作流程

1. **进行一轮搜索** - 使用SymbolicRegression进行符号回归搜索
2. **提取结果** - 获取Pareto前沿的最优表达式
3. **发送给LLM** - 将表达式（复杂度、损失、分数、方程）发送给LLM
4. **LLM选择** - LLM根据多个标准选择最有希望的表达式
5. **创建自定义种群** - 用LLM选择的表达式创建自定义初始种群
6. **继续搜索** - 使用自定义种群进行下一轮搜索
7. **循环** - 重复步骤1-6，直到达到最大迭代次数

## 安装依赖

在运行示例之前，需要安装HTTP和JSON包：

```julia
using Pkg
Pkg.add(["HTTP", "JSON"])
```

## 使用方法

### 基本用法

```julia
using SymbolicRegression
using SymbolicRegression: safe_pow

# 准备数据
X = randn(Float32, 3, 100)
y = Float32(2.0) * X[1, :] .+ Float32(3.0) * X[2, :] .^ 2 .- Float32(1.5)

# 设置选项
options = Options(;
    binary_operators=[+, *, -, /, safe_pow],
    unary_operators=[],
    population_size=50,
    maxsize=10,
)

# 运行LLM辅助的搜索
include("llm_integration.jl")

final_hof = symbolic_regression_with_llm(
    X, y,
    options;
    max_iterations=5,        # 总共5轮循环
    iterations_per_round=10,  # 每轮搜索10次迭代
    llm_selection_count=5,   # LLM每次选择5个表达式
    verbosity=1,              # 显示进度
)
```

### 运行示例文件

```bash
julia SymbolicRegression.jl/examples/llm_integration.jl
```

## 配置LLM

在 `llm_integration.jl` 文件中，你可以修改以下配置：

```julia
const MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"  # 模型名称
const API_BASE_URL = "https://ollama.cs.odu.edu"        # API地址
const API_KEY = "sk-704f0fe8734549c9b57ac09a69001ad7"   # API密钥
```

### 可用的模型

- `"meta-llama/Llama-3.1-8B-Instruct"`
- `"mistralai/Mistral-7B-Instruct-v0.3"`
- `"Qwen/Qwen3-32B"`
- `"openai/gpt-oss-120b"`

## 参数说明

### `symbolic_regression_with_llm` 函数参数

- `X::AbstractMatrix`: 输入数据矩阵
- `y::AbstractVector`: 目标值向量
- `options::AbstractOptions`: SymbolicRegression选项
- `max_iterations::Int=5`: 最大循环轮数
- `iterations_per_round::Int=10`: 每轮搜索的迭代次数
- `llm_selection_count::Int=5`: LLM每次选择的表达式数量（备用，如果LLM选择失败）
- `verbosity::Int=1`: 详细程度（0=静默，1=正常，2=详细）

## LLM Prompt 格式

发送给LLM的prompt包含：
- 当前轮数和总轮数
- 所有Pareto最优表达式的列表（复杂度、损失、分数、表达式）
- 选择标准说明

LLM需要返回JSON格式：
```json
{
  "selected_indices": [1, 3, 5],
  "reasoning": "选择这些表达式的原因..."
}
```

## 选择标准

LLM会根据以下标准选择表达式：
1. **损失(Loss)较低** - 表示拟合效果好
2. **复杂度适中** - 不要太简单（可能欠拟合）也不要太复杂（可能过拟合）
3. **表达式结构有潜力** - 可能通过变异和交叉产生更好的结果

## 注意事项

1. **API调用** - 每次循环都会调用LLM API，确保网络连接正常
2. **API密钥** - 确保API密钥有效且有足够的配额
3. **响应时间** - LLM响应可能需要几秒钟，请耐心等待
4. **错误处理** - 如果LLM调用失败，会使用前N个最佳表达式作为备用

## 示例输出

```
================================================================================
第 1/5 轮
================================================================================

[步骤1] 进行符号回归搜索...
找到 5 个Pareto最优表达式

当前最佳表达式:
  复杂度=1, 损失=23.74, 表达式=x1
  复杂度=3, 损失=13.57, 表达式=x2 * x2
  ...

[步骤2] 将结果发送给LLM进行选择...
LLM响应: {"selected_indices": [2, 3, 5], "reasoning": "..."}

[步骤3] 解析LLM响应并选择表达式...
LLM选择了索引: [2, 3, 5]

[步骤4] 创建自定义种群...
成功创建自定义种群，包含 3 个LLM选择的表达式和 47 个随机成员
```

## 故障排除

### 问题：LLM API调用失败

**解决方案**：
- 检查网络连接
- 验证API密钥是否正确
- 检查API服务是否可用

### 问题：LLM响应格式不正确

**解决方案**：
- 代码会自动尝试从文本中提取数字
- 如果完全失败，会使用前N个最佳表达式作为备用

### 问题：类型不匹配错误

**解决方案**：
- 确保使用本地开发版本的SymbolicRegression
- 检查所有成员是否使用相同的具体类型

## 扩展

你可以根据需要修改：
- LLM prompt格式
- 选择标准
- 种群创建策略
- 循环逻辑

## 与Python版本的对比

Python版本使用PySR，但PySR不支持自定义初始种群。Julia版本的优势：
- ✅ 支持自定义初始种群
- ✅ 更灵活的控制
- ✅ 更好的性能
- ✅ 原生Julia代码，无需Python-Julia桥接

