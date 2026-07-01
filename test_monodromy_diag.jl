using Pkg
Pkg.activate("/Users/yusuke/work/Teukolsky.jl")
using Teukolsky
using Printf

# Diagnose: ΔRup vs Rdown の漸近的ふるまいを確認
# ω = δ - iσ (虚軸の右側)
# Rup ~ Ctrans × r^3 × e^{iωr*} = Ctrans × r^3 × e^{σr*} e^{iδr*}  → 指数増大
# Rdown ~ r^{-1} e^{-iωr*} = r^{-1} e^{-σr*} e^{-iδr*}              → 指数減衰
# ΔRup ∝ e^{σr*} → Rdownと同じ桁数になり得ない

s, l, m, a = -2, 2, 2, 0.0
σ = 0.3
δ = 1e-6
ω_R = complex(δ, -σ)

ν, p = compute_nu(s, l, m, a, ω_R)
fn = compute_fn(p, ν; nmax=60)
amp = compute_amplitudes(s, l, m, a, ω_R; nmax=60)

q_info = compute_q(s, l, m, a, ω_R; nmax=60)

println("="^80)
println("漸近的ふるまいの確認: ω = $δ - $(σ)i")
println("="^80)
@printf("%-5s  %-14s  %-14s  %-14s  %-14s  %-14s\n",
        "r", "|Rup|", "|Rdown|", "|Rin|", "|ΔRup|", "|ΔRup/Rdown|")
println("-"^80)

for r in [4.0, 6.0, 8.0, 10.0, 12.0, 15.0]
    Rup_R = Rup(p, ν, fn, r; nmax=60)
    Rdown_R = Rdown(p, ν, fn, r; nmax=60)
    Rin_R = Rin(p, ν, fn, r; nmax=60)

    # ΔRup = conj(Rup) - Rup = -2i Im(Rup)  (Schwarzschild, a=0)
    ΔRup = conj(Rup_R) - Rup_R

    @printf("r=%4.0f  %14.4e  %14.4e  %14.4e  %14.4e  %14.4e\n",
            r, abs(Rup_R), abs(Rdown_R), abs(Rin_R), abs(ΔRup), abs(ΔRup/Rdown_R))
end

println()
println("="^80)
println("ΔRup = -2i Im(Rup) は指数増大, Rdown は指数減衰")
println("→ ΔRup = K × Rdown (K: r非依存) は成立し得ない")
println()
println("代わりに ΔRup / Rin を見る:")
println("="^80)
@printf("%-5s  %-30s  %-30s\n", "r", "ΔRup/Rin", "|ΔRup/Rin|")
println("-"^80)

for r in [4.0, 6.0, 8.0, 10.0, 12.0, 15.0]
    Rup_R = Rup(p, ν, fn, r; nmax=60)
    Rin_R = Rin(p, ν, fn, r; nmax=60)
    ΔRup = conj(Rup_R) - Rup_R
    ratio = ΔRup / Rin_R
    @printf("r=%4.0f  %+.8e %+.8ei  %.4e\n",
            r, real(ratio), imag(ratio), abs(ratio))
end

# Greenの不連続からqの検証を試みる
# G = Rin(r<) × Rup(r>) / (2iω × W)
# ΔG = Rin × ΔRup / (2iω × W)  (Rinは不連続なし、Wは不連続なし と仮定)
# ただし実際にはW自体も不連続を持つ可能性がある

println()
println("="^80)
println("Green関数の不連続: ΔG = Rin × ΔRup / (2iω W)")
println("W[Rin, Rup] の確認")
println("="^80)

δr = 1e-7
for r in [4.0, 6.0, 8.0, 10.0]
    Δ_BL = r^2 - 2r + a^2
    Rin_R = Rin(p, ν, fn, r; nmax=60)
    Rup_R = Rup(p, ν, fn, r; nmax=60)
    dRin = (Rin(p, ν, fn, r+δr; nmax=60) - Rin(p, ν, fn, r-δr; nmax=60)) / (2δr)
    dRup = (Rup(p, ν, fn, r+δr; nmax=60) - Rup(p, ν, fn, r-δr; nmax=60)) / (2δr)
    W = Rin_R * dRup - Rup_R * dRin
    @printf("r=%4.0f  W = %+.6e %+.6ei  W/Δ = %+.6e %+.6ei\n",
            r, real(W), imag(W), real(W/Δ_BL), imag(W/Δ_BL))
end

# Btrans, Ctrans の確認
println()
println("Binc  = $(amp.Binc)")
println("Bref  = $(amp.Bref)")
println("Ctrans= $(amp.Ctrans)")
println("|q|   = $(abs(q_info.q))")
println("q     = $(q_info.q)")
