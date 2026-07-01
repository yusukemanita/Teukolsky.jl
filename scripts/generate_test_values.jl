"""
テスト用参照値を生成するスクリプト。

バグ修正前後での数値の変化を確認するために使用する。
結果を test/reference_values.jl に書き出す。
"""

using Pkg
Pkg.activate("/Users/yusuke/work/Teukolsky.jl")
using Teukolsky
using Dates
using Printf

io = open(joinpath(@__DIR__, "test", "reference_values.jl"), "w")

function emit(io, varname, val::ComplexF64)
    println(io, "$(varname) = $(repr(val))")
end
function emit(io, varname, val::Float64)
    println(io, "$(varname) = $(repr(val))")
end

println(io, "# 自動生成: generate_test_values.jl")
println(io, "# コミット: ", read(`git rev-parse --short HEAD`, String) |> strip)
println(io, "# 日時: ", string(now()))
println(io)

cases = [
    # (s, l, m, a,  ω,       label)
    (-2, 2, 2, 0.0, 0.3,    "sch_s2_l2_m2_om03"),
    (-2, 2, 2, 0.0, 0.5,    "sch_s2_l2_m2_om05"),
    (-2, 2, 2, 0.0, 1.0,    "sch_s2_l2_m2_om10"),
    (-2, 2, 2, 0.5, 0.3,    "ker05_s2_l2_m2_om03"),
    (-2, 2, 2, 0.9, 0.3,    "ker09_s2_l2_m2_om03"),
    (-2, 2, 2, 0.9, 0.5,    "ker09_s2_l2_m2_om05"),
    ( 0, 2, 0, 0.0, 0.3,    "sch_s0_l2_m0_om03"),
]

for (s, l, m, a, ω, lbl) in cases
    println("計算中: s=$s l=$l m=$m a=$a ω=$ω ...")
    ν, p = compute_nu(s, l, m, a, ω)
    amp  = compute_amplitudes(s, l, m, a, ω)

    println(io, "# s=$s, l=$l, m=$m, a=$a, ω=$ω")
    emit(io, "nu_$(lbl)",   ComplexF64(ν))
    emit(io, "Binc_$(lbl)", ComplexF64(amp.Binc))
    emit(io, "Bref_$(lbl)", ComplexF64(amp.Bref))
    emit(io, "lambda_$(lbl)", ComplexF64(p.λ))
    println(io)
end

close(io)
println("test/reference_values.jl を書き出しました")
