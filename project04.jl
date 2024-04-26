#=
利用谱求导矩阵的一维定态Schrodinger方程求解器:
1、采用原子单位制(a.u.)，定态Schrodinger方程形式为:  -ħ²/2m*▽·(▽ψ) + V*ψ = E*ψ, ħ=1
2、由于谱求导矩阵基于DFT得到，求解边界为周期性边界，势函数V将看作V的周期延拓
=#
using LinearAlgebra, ToeplitzMatrices
using GLMakie
const ħ = 1.0
const m = 1.0  # 电子

# 主函数
function main()
    # 离散设置
    L = 50  # 求解区域长度
    N = 512  # 节点数(偶数)
    x = L/N*(-N/2:N/2-1)  # 求解区域

    # 势函数
    a = 5.0
    M = 2
    Vmin = -4
    V = zeros(N)
    for m in -M:M
        V += Vmin*exp.(-(x .- m * a).^2)
    end

    # 构造二阶谱求导矩阵D₂
    h = 2π/N
    C = (2π / L)^2  # 放缩系数
    vc₀ = [1/2 * (-1)^(j+1) * (csc(j*h/2))^2 for j=1:N-1]
    vc = [-π^2/(3*h^2)-1/6; vc₀]  # D₂矩阵的第一列
    D₂ = (2π/L)^2*SymmetricToeplitz(vc)  # Toeplitz矩阵

    # 构建Hamilton矩阵Hami
    Hami = -ħ^2/(2*m) * D₂ + diagm(V)

    # 求解N个本征值E与各点本征函数值ψ
    S = eigen(Hami)
    Val = S.values  # 本征值
    Vec = S.vectors  # 本征向量

    # 本征函数归一化
    Vec2 = abs2.(Vec)  # |ψ|²
    C = sqrt.(1 ./(sum(L/N*Vec2, dims=1)))  # 归一化系数
    for j=1:N
        Vec[:, j] = Vec[:, j]*C[j]
    end

    # 绘图(势场，能量，概率密度)
    Vinf = maximum([V[1], V[end]]) 

    fig = Figure()
    ax1 = Axis(fig[1, 1], xlabel="x (a.u.)", ylabel="Energy (a.u.)", title="V(x)")
    lines!(ax1, x, V, linewidth=2)

    ax2 = Axis(fig[2, 1], xlabel="x", ylabel="ψ(x)")
    for i in 1:20  # 最多到N
        if Vmin < Val[i] < Vinf  # 束缚态条件
            nLabel = string("n = ", i)
            lines!(ax2, x, Vec[:, i], linewidth=2, label=nLabel)
        else
            lines!(ax2, x, Vec[:, i], lineswidth=4, linestyle=:dashdot)
        end
    end
    GLMakie.Legend(fig[2, 2], ax2, "束缚态")

    ax3 = Axis(fig[3, 1], xlabel="解序号n", ylabel="Eₙ")
    for i in 1:40  # 最多到N
        if Vmin < Val[i] < Vinf 
            GLMakie.scatter!(ax3, (i, Val[i]), color=:red)
        else
            GLMakie.scatter!(ax3, (i, Val[i]), color=:blue)
        end
    end
    display(fig)
end
main()
