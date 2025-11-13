"""
直接加载本地修改的 SymbolicRegression 模块（绕过包管理器）

这个方法不需要使用 Pkg.develop()，避免了依赖冲突问题。
"""

# 方法：直接 include 本地文件
# 将本地 src 目录添加到 LOAD_PATH
const LOCAL_SR_PATH = joinpath(@__DIR__, "..", "src")
pushfirst!(LOAD_PATH, LOCAL_SR_PATH)

# 现在可以加载本地版本
# 注意：需要先移除已加载的 SymbolicRegression（如果已加载）
if isdefined(Main, :SymbolicRegression)
    @warn "SymbolicRegression 已经加载，可能需要重启 Julia 才能使用本地版本"
end

# 加载本地版本
include(joinpath(LOCAL_SR_PATH, "SymbolicRegression.jl"))

# 验证是否使用了本地版本
println("SymbolicRegression 路径: ", pathof(SymbolicRegression))
println("期望路径包含: ", LOCAL_SR_PATH)

