#= 
1、利用时间演化算子高精度求解一维Schrödinger方程，几乎支持任意时间步长；其中Hamilton算子不显含时
2、采用原子单位制(a.u.)，有：
长度(SI) = 玻尔半径(SI) * x(a.u.)，玻尔半径为5.2917721067e-11m
时间(SI) = 2.418884326e-17s * t(a.u.)
速度(SI) = 2.187691263207598e6m/s * v(a.u.)
质量(SI) = 电子质量(SI) * m(a.u.)，电子质量为9.10938215e-31kg
动量(SI) = 1.9928515742774247e-24kg*m/s * P(a.u.)
能量(SI) = 4.3597446499e-18J * E(a.u.)
角动量(SI) = ħ(SI) * L(a.u.), ħ(SI) = 1.0545718176e-34kg*m^2/s
3、采用第一类边界条件，边界点外为无穷势垒
=#
using GLMakie
set_theme!(theme_dark())
using LinearAlgebra

const ħ = 1.0
const m = 1.0  # 电子
Δt = 0.1  # 时间步长
xmax = 20.0  # x轴最大值
numT = 200  # 时间步数
Nx = 500  # 离散节点数

# 势函数(不显含时)
function Vfun(X::StepRangeLen)
    n = 3
    V = n/sqrt(π) * (
        exp.(-n^2 * X.^2) .+
        exp.(-n^2 * (X .- xmax/5).^2) .+
        exp.(-n^2 * (X .- 2xmax/5).^2) .+
        exp.(-n^2 * (X .- 3xmax/5).^2) .+
        exp.(-n^2 * (X .- 4xmax/5).^2)
    )
    V[1] = 1e8
    V[end] = 1e8
    return V
end

# Schrödinger方程求解
function Qsolver(Δt, xmax, numT::Int64, Nx::Int64)
    # 离散设置
    X = range(-xmax, xmax, Nx)
    Δx = abs(X[end] - X[end-1])
    # Δt时间演化算子U
    V = Matrix{Float64}(I, Nx, Nx) .* Vfun(X)
    D = SymTridiagonal(fill(-2., Nx), fill(1., Nx-1))
    U = ℯ^(im * ħ * Δt / (2m * Δx^2) * D + Δt * V / (im * ħ))
    # 初始分布(高斯波包)
    Ψ = zeros(ComplexF64, Nx, numT)
    δₓ = xmax / 40  # Gauss波包位置标准差
    x₀ = -xmax / 2  # 波包中心初始位置
    p₀ = 3.0  # 初始动量
    Ψ[:, 1] = 1/(2π*δₓ^2)^(1/4) * exp.(-((X .- x₀) / 2δₓ).^2) .* exp.(im * (p₀ * X / ħ))
    # 求解
    for k in 2:numT
        Ψ[:, k] = U * Ψ[:, k-1]
        # 第一类边界条件
        Ψ[1] = 0.0 + 0.0 * im
        Ψ[end] = 0.0 + 0.0 * im
    end
    ΨAbs2 = abs2.(Ψ)

    return Ψ, ΨAbs2, X
end
Ψ, ΨAbs2, X = @time Qsolver(Δt, xmax, numT, Nx)

# 动画
function video(Δt, X, numT::Int64, ΨAbs2::Matrix{Float64})
    V = Vfun(X)
    Δx = abs(X[2] - X[1])
    val = [Δx * sum(ΨAbs2[:, k]) for k in 1:numT]  # 概率积分，检验守恒性

    k = Observable(1)
    time = @lift round(($k - 1) * Δt, digits=2)
    err = @lift (val[$k] - val[1]) / val[1]
    Ob = @lift ΨAbs2[:, $k]

    fig = Figure()
    tit = @lift string("时间 t = ", $time, " (a.u.)\n", "守恒性相对误差 err = ", $err)
    ax1 = Axis(fig[1, 1], title=tit, xlabel="x (a.u.)", ylabel="概率密度值", yticklabelcolor=:blue)
    ax2 = Axis(fig[1, 1], ylabel="E (a.u.)", yticklabelcolor=:red, yaxisposition=:right)
    hidexdecorations!(ax2)
    l1 = lines!(
        ax1, X, Ob,
        linewidth=2,
        color=:blue
    )
    l2 = lines!(
        ax2, X[2:end-1], V[2:end-1],
        linewidth=0.5,
        color=:red
    )
    Legend(fig[1, 2], [l1, l2], ["|Ψ(x, t)|²", "V(x)"])
    # 动画迭代记录并绘制
    timesteps = 1:numT  # k执行序列
    framerate = 15  # 帧率
    record(fig, "ComPhy//1D_Schrodinger02.mp4", timesteps; framerate) do i
        k[] = i
    end
end
video(Δt, X, numT, ΨAbs2)
