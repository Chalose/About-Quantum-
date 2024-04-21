#= 绘制氢原子各本征态电子云图
令玻尔半径 a = 1
=#
using GLMakie
set_theme!(theme_black())
using AssociatedLegendrePolynomials  # 提供连带勒让德函数
using SpecialFunctions  # 提供伽马函数

# 连带拉盖尔多项式associated Laguerre
function Lun(u::Int, n::Int, x)
    res = 0.0
    c = gamma(u + n + 1)
    for k in 0:n
        res += (-x)^k * c / (factorial(k) * factorial(n - k) * gamma(u + k + 1))
    end
    return res
end
# 本征波函数Ψ
function Ψ(A, B, n::Int, l::Int, m::Int, r, θ, ϕ)
    B * Lun(2l + 1, n - l - 1, 2r / n) * exp(-r / n) * (2r / n)^l * A * Plm(l, m, cos(θ)) * exp(im * ϕ)
end
# 直角坐标转化为球坐标
function rt(x, y, z)
    sqrt(x.^2 + y.^2 + z.^2)
end
function θt(x, y, z)
    acos(z / sqrt(x.^2 + y.^2 + z.^2))
end
function ϕt(x, y, z)
    atan(y / x)
end
# 绘图======================================================================================================
function draw(n::Int, l::Int, m::Int) # n为主量子数(1,2,...)；l为角量子数(0,1,2,...,n-1)；m为磁量子数(0,...,l-1,l 此处取大于0的m值)
    # 范围
    N = 100
    ls = range(-80, 80, N)
    xs = ls; ys = ls; zs = ls
    # 系数
    A = Nlm(l, m)  # 连带勒让德函数的归一化系数, 球谐函数Yₗₘ = Nlm * Plm * exp(im * ϕ)
    B = sqrt((2 / n)^3 * factorial(n - l - 1)/(2n * factorial(n + l)))
    # 本征波函数Ψ
    Ψval = [Ψ(A, B, n, l, m, rt(x, y, z), θt(x, y, z), ϕt(x, y, z)) for x in xs, y in ys, z in zs]  # 概率幅
    AbsΨval = abs.(Ψval)  # 概率密度
    # 绘图
    Tit = string("|", n, ",", l, ",", m, ">")
    fig = Figure()
    ax1 = Axis3(fig[1, 1], title=Tit, xlabel="x / a", ylabel="y / a", zlabel="z / a", aspect=(1, 1, 1))
    v1 = contour!(
        ax1, xs, ys, zs, AbsΨval,
        colormap=:gnuplot,
        levels=4,
        alpha=0.3
    )
    v2 = volume!(
        ax1, xs, ys, zs, AbsΨval,
        colormap=:turbo,
    )
    Colorbar(fig[1, 2], v1)
    Colorbar(fig[1, 3], v2, label="Probability density")
    display(fig)
    save("Hydrogen02.png", fig)
end

draw(6, 4, 1)
