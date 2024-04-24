#= 
1、通过Crank-Nicholson法求解二维Schrödinger方程(理论上也适用于含时Hamilton算子)，具有良好的求解稳定性
2、采用原子单位制(a.u.)，有：
长度(SI) = 玻尔半径(SI) * x(a.u.)，玻尔半径为5.2917721067e-11m
时间(SI) = 2.418884326e-17s * t(a.u.)
速度(SI) = 2.187691263207598e6m/s * v(a.u.)
质量(SI) = 电子质量(SI) * m(a.u.)，电子质量为9.10938215e-31kg
动量(SI) = 1.9928515742774247e-24kg*m/s * P(a.u.)
能量(SI) = 4.3597446499e-18J * E(a.u.)
角动量(SI) = ħ(SI) * L(a.u.), ħ(SI) = 1.0545718176e-34kg*m^2/s
3、采用第一类边界条件，边界点外为无穷势垒
4、由于采取隐式求解，线性方程组Ax=b涉及大型对称稀疏复矩阵求解，当前使用左除"\"求解，尚未采用更高效的求解器
=#
using GLMakie
set_theme!(theme_black())
using LinearAlgebra, SparseArrays

const ħ = 1.0
const m = 1.0  # 电子
Δt = 0.005  # 时间步长
xmax = 5  # x轴最大值
ymax = 2.5  # y轴最大值
numT = 400  # 时间步数
Nx = 200  # x方向节点数
Ny = 100  # y方向节点数

# 势函数矩阵
function Vfun(X::StepRangeLen, Y::StepRangeLen)
    max = 1e8
    #V = [5 * x^2 + 5 * y^2 for x in X, y in Y]
    # 双缝
    a = 8  # 缝宽格子数
    b = 3  # 势垒厚度格子数
    c = 16  # 中央挡墙宽度格子数
    h = Int((Ny - 2a - c) / 2)
    V = zeros(Ny, Nx)
    V[1:h, Int(Nx/2):Int(Nx/2 + b)] .= max
    V[h+a:h+a+c, Int(Nx/2):Int(Nx/2 + b)] .= max
    V[h+2a+c:end, Int(Nx/2):Int(Nx/2 + b)] .= max
    # 拟无穷势垒边界
    V[:, 1] .= max
    V[:, end] .= max
    V[1, :] .= max
    V[end, :] .= max

    return V
end

# 线性方程组求解函数
function LinSolver!(ΨAbs2, A, M, VΨ, numT, Ny, Nx)
    for k in 2:numT
        b = M * VΨ
        VΨ = A \ b
        ΨAbs2[:, :, k] = reshape(abs2.(VΨ), Ny, Nx)
    end
end

# 2维Schrödinger方程求解
function Qsolver(Δt, xmax, ymax, numT, Nx::Int, Ny::Int)
    # 离散设置
    X = range(-xmax, xmax, Nx)
    Y = range(-ymax, ymax, Ny)
    Δx = abs(X[end] - X[end-1])
    Δy = abs(Y[end] - Y[end-1])
    # 初始分布
    δ = xmax / 5
    x₀ = -xmax / 2
    y₀ = 0.0
    p₀ = 10
    Ψ₀ = [1/(δ*sqrt(π)) * exp(-((x - x₀)^2 + (y - y₀)^2) / (2*δ^2)) * exp(-im * p₀ * x / ħ) for x in X, y in Y]'  # 注意转置，且保证波包不要过于靠近边界破坏归一化及边界条件
    VΨ = reshape(Ψ₀, Ny*Nx, 1)  # 波函数矩阵向量化
    # 算子构建
    Rx = im * ħ * Δt / (4m * (Δx)^2)
    Ry = im * ħ * Δt / (4m * (Δy)^2)
    Rv = im * Δt / (2ħ)
    UM = sparse(Matrix{ComplexF64}(I, Ny*Nx, Ny*Nx))
    UMx = sparse(Matrix{ComplexF64}(I, Ny, Ny))
    UMy = sparse(Matrix{ComplexF64}(I, Nx, Nx))
    Dx = sparse(SymTridiagonal(fill(0.0+0.0im, Nx), fill(Rx, Nx-1)))
    Dy = sparse(SymTridiagonal(fill(0.0+0.0im, Ny), fill(Ry, Ny-1)))
    V = reshape(Vfun(X, Y), Ny*Nx, 1)  # 势函数矩阵向量化
    Dᵥ₁ = sparse(UM .* (1 .+ 2Rx .+ 2Ry .+ Rv * V))
    Dᵥ₂ = sparse(UM .* (1 .- 2Rx .- 2Ry .- Rv * V))
    G = sparse(kron(Dx, UMx) + kron(UMy, Dy))
    A = sparse(-G + Dᵥ₁)  # 系数矩阵A，有线性方程组: A * VΨ(k+1 step) = b
    M = sparse(G + Dᵥ₂)  # b = M * VΨ(k step)
    # 线性方程组 Ax=b 求解
    ΨAbs2 = zeros(Ny, Nx, numT)  # 存储概率密度
    ΨAbs2[:, :, 1] = abs2.(Ψ₀)
    LinSolver!(ΨAbs2, A, M, VΨ, numT, Ny, Nx)

    return ΨAbs2, X, Y, V
end
ΨAbs2, X, Y, V = @time Qsolver(Δt, xmax, ymax, numT, Nx, Ny)

# 动画
function video(Δt, X, Y, numT, ΨAbs2, V)
    Δx = abs(X[end] - X[end-1])
    Δy = abs(Y[end] - Y[end-1])
    val = [Δx * Δy * sum(ΨAbs2[:, :, k]) for k in 1:numT]  # 概率积分，检验守恒性
    VMatrix = reshape(V, Ny, Nx)
    Vshow = VMatrix[2:Ny-1, 2:Nx-1]'  # 设置双缝时，此处加转置
    
    k = Observable(1)
    time = @lift round(($k - 1) * Δt, digits=5)
    err = @lift (val[$k] - val[1]) / val[1]
    Ob = @lift ΨAbs2[:, :, $k]'

    fig = Figure(size=(500, 550))
    tit = @lift string("时间 t = ", $time, " (a.u.)\n", "守恒性相对误差 err = ", $err)
    ax1 = Axis(fig[1, 1][1, 1], title="位势 V(x, y)", xlabel="x (a.u.)", ylabel="y (a.u.)", aspect=xmax/ymax)
    h1 = heatmap!(
        ax1, X[2:end-1], Y[2:end-1], Vshow,
        interpolate=true,
    )
    Colorbar(fig[1, 1][1, 2], h1, label="Energy (a.u.)")

    ax2 = Axis(fig[2, 1][1, 1], title=tit, xlabel="x (a.u.)", ylabel="y (a.u.)", aspect=xmax/ymax)
    h2 = heatmap!(
        ax2, X, Y, Ob,
        interpolate=true,
        colormap=:vik
    )
    Colorbar(fig[2, 1][1, 2], h2, label="|Ψ|²")
    # 动画迭代记录并绘制
    timesteps = 1:numT  # k执行序列
    framerate = 24  # 帧率
    record(fig, "ComPhy//2D_Schrodinger.mp4", timesteps; framerate) do i
        k[] = i
    end
end
video(Δt, X, Y, numT, ΨAbs2, V)
