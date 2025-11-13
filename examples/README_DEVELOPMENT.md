# 使用本地开发版本

如果你修改了 SymbolicRegression.jl 的源代码，你需要让 Julia 使用本地开发版本而不是已安装的包版本。

## 方法1: 在 Julia REPL 中激活本地环境

```julia
# 在 Julia REPL 中
using Pkg
Pkg.develop(path="C:/Users/cs_chu034/download/projects/SR_LLM/SymbolicRegression.jl")
# 或者使用相对路径
# Pkg.develop(path="./SymbolicRegression.jl")

# 然后重新加载模块
using SymbolicRegression
```

## 方法2: 在脚本开头设置环境

在运行示例文件之前，在 Julia REPL 中执行：

```julia
using Pkg
Pkg.activate("C:/Users/cs_chu034/download/projects/SR_LLM/SymbolicRegression.jl")
# 或者
# Pkg.activate("./SymbolicRegression.jl")
```

## 方法3: 使用本地路径直接加载

修改示例文件，在开头添加：

```julia
# 将本地路径添加到 LOAD_PATH
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

# 然后加载模块
using SymbolicRegression
```

## 方法4: 在项目根目录创建 Manifest.toml

如果你在项目根目录（SR_LLM），可以创建一个 `Project.toml` 和 `Manifest.toml`：

```julia
using Pkg
Pkg.activate(".")
Pkg.develop(path="./SymbolicRegression.jl")
```

## 验证是否使用了本地版本

运行以下代码检查模块路径：

```julia
using SymbolicRegression
println(pathof(SymbolicRegression))
# 应该显示本地路径，而不是 .julia/packages/ 下的路径
```

如果显示的是 `.julia/packages/` 下的路径，说明仍在使用已安装的版本。

## 常见问题

### 问题：修改代码后没有生效

**解决方案**：
1. 重启 Julia REPL
2. 重新加载模块：`using SymbolicRegression` 或 `import Pkg; Pkg.resolve()`
3. 确保使用的是本地开发版本

### 问题：找不到模块

**解决方案**：
确保 `SymbolicRegression.jl` 目录下有 `Project.toml` 文件，并且结构正确。

