#=
采用Metropolis采样对2维Ising model作多线程蒙特卡洛模拟，正方晶格有 行数*列数：Ny * Nx;
H = sum_<ij>( Jz*σzi*σzj ) - h*sum_i( σzi )  , 周期性边界；
当外场h为小值时，求解的收敛性更好(怀疑是外场破坏了体系的简并态，即相同能量但具有不同的总磁矩)

采样规则：
1、一次Metropolis采样只尝试翻转一个格点自旋，遍历全部格点作为一个MCstep，完成一个时间变量time(初始为1)；
2、一次MCstep完成后求取一次体系总能E，磁矩m，即time时刻瞬时物理量；
3、执行完timeMax次MCstep，即总共进行timeMax * Ny * Nx次Metropolis抽样，进行期望与方差计算得到宏观热力学量.
=#
using Statistics, Distributions
using GLMakie
set_theme!(theme_black())
using Base.Threads
println("当前线程总数: ", Threads.nthreads())
# 全局参数 --------------------------------------------------------------------------------------------
Ny = 30  # 晶格行数
Nx = 30  # 晶格列数
N = Ny * Nx
Jz = -1  # 交换积分(<0 铁磁；>0 反铁磁)
h = 0.0  # 外磁场作用
kb = 1.0  # Boltzmann常数
timeMax = 15000 # MCstep步数(对零场h=0，timeMax应最好达到10⁴数量级)
Nsam = 5000  # 某T下平衡态采样数
lenT = 32  # 温度T采样点数
TE = 2.27  # 温度T正态采样期望

# 生成温度T的正态采样序列TR
function TRange()
    σs = 0.9  # 方差取0~1内
    TR = sort(rand(Normal(TE, σs), lenT))
    while minimum(TR) <= 0
        TR = sort(rand(Normal(TE, σs), lenT))
    end
    return TR
end
TR = TRange()
Tmin = TR[1]  # 最低温度
Tmid = TR[Int(round(lenT/2))]  # 中间温度
Tmax = TR[end]  # 最高温度
println("Tmin = ", Tmin, " Tmid = ", Tmid, " Tmax = ", Tmax)
println("loading...")

# 调用函数 --------------------------------------------------------------------------------------------
# 自旋flip点的全部相邻格点线性索引
function AdjoinIndex(k::Int)
    G = CartesianIndices((1:Ny, 1:Nx))[k]
    i = G[1]
    j = G[2]
    kU = (j - 1) * Ny + (i + Ny - 2) % Ny + 1  # up
    kD = (j - 1) * Ny + i % Ny + 1  # down
    kL = (j + Nx - 2) % Nx * Ny + i  # left
    kR = j % Nx * Ny + i  # right

    return kU, kD, kL, kR
end

# 遍历晶格的Metropolis采样
function MCstep!(σMesh::Matrix, T)
    # 自旋网格遍历
    for k in eachindex(σMesh)
        kU, kD, kL, kR = AdjoinIndex(k)
        ΔE = -2 * σMesh[k] * Jz * (σMesh[kU] + σMesh[kD] + σMesh[kL] + σMesh[kR]) + 2h * σMesh[k]
        r = exp(-ΔE / (kb * T))  # 接受新组态概率(Boltzmann因子)
        p = rand()
        if r > 1 || (r < 1 && p < r)
            σMesh[k] = -σMesh[k]
        end
    end

    # 计算总磁矩M，总能E
    M = sum(σMesh)
    E = -h * M
    for k in eachindex(σMesh)
        kU, kD, kL, kR = AdjoinIndex(k)
        E += Jz/2 * σMesh[k] * (σMesh[kU] + σMesh[kD] + σMesh[kL] + σMesh[kR])  # 重复计算除以2
    end

    return M, E
end

# 对某温度T的马尔可夫链过程
function Markov(T)
    # 自旋网格初始化
    if T < 1.5 && Jz < 0
        σMesh = ones(Ny, Nx)
    else
        σMesh = rand(-1:2:1, Ny, Nx)
    end

    # MarkovChain
    Mt = zeros(Nsam)  # 总磁矩序列( time = (timeMax - Nsam) + 1 ~ timeMax )
    Et = zeros(Nsam)  # 总能量序列
    k = 1
    for time in 1:timeMax
        M, E = MCstep!(σMesh, T)
        # 采样
        if time >= (timeMax - Nsam) + 1
            Mt[k] = M
            Et[k] = E
            k += 1
        end
    end

    # 当前温度下的宏观热力学量
    Average_U = mean(Et) / N  # 平均内能U / N
    Average_M = mean(abs.(Mt)) / N  # 磁化强度绝对值<|M|> / N
    C = 1/N * 1/(kb * T^2) * (mean(Et.^2) - (N * Average_U)^2)  # 定体比热Cᵥ / N
    χ = 1/N * 1/(kb * T) * (mean(Mt.^2) - (N * Average_M)^2)  # 磁化率χ

    return Average_U, Average_M, C, χ, σMesh
end

# 主函数 --------------------------------------------------------------------------------------------
function main()
    # 建立字典变量，存储不同线程处理的结果；  T => value
    dict_U = Dict()
    dict_M = Dict()
    dict_C = Dict()
    dict_χ = Dict()
    # 部分自旋构型
    σMesh_Tmin = zeros(Ny, Nx)
    σMesh_Tmid = zeros(Ny, Nx)
    σMesh_Tmax = zeros(Ny, Nx)

    # 多线程计算不同温度下的物理量
    @time @sync for i in 1:lenT
        Threads.@spawn begin
            T = TR[i]
            dict_U[T], dict_M[T], dict_C[T], dict_χ[T], σMesh = Markov(T)
            # 记录部分自旋构型
            if T == Tmin
                σMesh_Tmin = σMesh
            elseif T == Tmid
                σMesh_Tmid = σMesh
            elseif T == Tmax
                σMesh_Tmax = σMesh
            end
        end
    end

    # 后处理
    U_T = zeros(lenT)
    M_T = zeros(lenT)
    C_T = zeros(lenT)
    χ_T = zeros(lenT)
    for i in 1:lenT
        T = TR[i]
        U_T[i] = dict_U[T]
        M_T[i] = dict_M[T]
        C_T[i] = dict_C[T]
        χ_T[i] = dict_χ[T]
    end

    return U_T, M_T, C_T, χ_T, σMesh_Tmin, σMesh_Tmid, σMesh_Tmax
end
U_T, M_T, C_T, χ_T, σMesh_Tmin, σMesh_Tmid, σMesh_Tmax = main()

function draw()
    f1 = Figure()
    tit = string("Ny = ", Ny, " Nx = ", Nx, " Jz = ", Jz, " h = ", h, " kB = ", kb)
    ax1 = Axis(f1[1, 1], xlabel=L"T", ylabel=L"\frac{U(T)}{N}", title=tit)
    scatter!(ax1, TR, U_T)  # 平均内能U——温度T
    ax2 = Axis(f1[2, 1], xlabel=L"T", ylabel=L"\frac{C_v(T)}{N}")
    scatter!(ax2, TR, C_T)  # 比热Cᵥ/N——温度T
    display(GLMakie.Screen(), f1)

    f2 = Figure()
    ax1 = Axis(f2[1, 1], xlabel=L"T", ylabel=L"\frac{⟨|m|⟩}{N}", title=tit)
    scatter!(ax1, TR, M_T, color=:yellow)  # 磁化强度绝对值<|M|> / N——温度T
    ax2 = Axis(f2[2, 1], xlabel=L"T", ylabel=L"χ(T)")
    scatter!(ax2, TR, χ_T, color=:yellow)  # 磁化率χ——温度T
    display(GLMakie.Screen(), f2)

    f3 = Figure()
    lab1 = string("T = ", round(Tmin, digits=2))
    ax1 = Axis(f3[1, 1], aspect=Nx/Ny, xlabel=lab1)
    heatmap!(ax1, σMesh_Tmin', colorrange=(-1, 1))
    lab2 = string("T = ", round(Tmid, digits=2), "\n", tit)
    ax2 = Axis(f3[1, 2], aspect=Nx/Ny, xlabel=lab2)
    heatmap!(ax2, σMesh_Tmid', colorrange=(-1, 1))
    lab3 = string("T = ", round(Tmax, digits=2))
    ax3 = Axis(f3[1, 3], aspect=Nx/Ny, xlabel=lab3)
    heatmap!(ax3, σMesh_Tmax', colorrange=(-1, 1))
    display(GLMakie.Screen(), f3)
end
draw()
