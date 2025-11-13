# 环境设置指南

## 问题：缺少依赖包（如 DispatchDoctor）

当使用本地开发版本的 SymbolicRegression 时，需要先安装所有项目依赖。

## 解决方案

### 方法1：使用 test.jl（推荐）

`test.jl` 文件已经包含了自动设置环境的代码：

```bash
julia SymbolicRegression.jl/examples/test.jl
```

这个脚本会：
1. 激活 SymbolicRegression 项目
2. 运行 `Pkg.instantiate()` 安装所有依赖（包括 DispatchDoctor）
3. 安装 HTTP 和 JSON 包
4. 运行 LLM 集成示例

### 方法2：手动设置

在 Julia REPL 中：

```julia
using Pkg

# 1. 激活项目
Pkg.activate("SymbolicRegression.jl")

# 2. 安装所有依赖
Pkg.instantiate()

# 3. 安装额外依赖
Pkg.add(["HTTP", "JSON"])

# 4. 运行示例
include("SymbolicRegression.jl/examples/llm_integration.jl")
```

### 方法3：使用 Pkg.develop（用于开发）

如果你想在全局环境中使用本地开发版本：

```julia
using Pkg

# 开发模式安装（链接到本地目录）
Pkg.develop(path="SymbolicRegression.jl")

# 安装依赖
Pkg.instantiate()

# 安装额外依赖
Pkg.add(["HTTP", "JSON"])
```

## 常见错误

### 错误：`Package DispatchDoctor not found`

**原因**：项目依赖未安装

**解决**：运行 `Pkg.instantiate()` 在项目目录中

### 错误：`Package SymbolicRegression not found`

**原因**：未激活项目或未使用开发模式

**解决**：
- 使用 `Pkg.activate("SymbolicRegression.jl")` 激活项目
- 或使用 `Pkg.develop(path="SymbolicRegression.jl")` 开发模式安装

## 验证安装

运行测试脚本验证环境是否正确设置：

```bash
julia SymbolicRegression.jl/examples/test_llm_integration.jl
```

如果看到 "✓ 测试通过！LLM API可以正常使用。"，说明环境设置成功。

