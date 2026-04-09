using Pkg
Pkg.activate("/Users/yusuke/work/BHPtoolkit.jl")

using BHPtoolkit
using Printf

# ============================================================
#  ブランチカット不連続量 ΔG(σ) の2つの計算法を比較
#
#  法1 (現行, BigFloat): 両側からアプローチ
#    G_R = G(+δ - iσ, m)
#    G_L = conj(G(+δ + iσ, -m))  [対称性を使用]
#    ΔG  = G_R - G_L
#
#  法2 (モノドロミー, Float64): ω = -iσ でν と -ν-1 の2ブランチ
#    ブランチカットを跨ぐとき ν → -ν-1 が起きる。
#    同一点 ω = -iσ で
#    G_ν     = Bref(ν)/(2iω Binc(ν))       with Kν(ν), Kνn(-ν-1)
#    G_{-ν-1}= Bref(-ν-1)/(2iω Binc(-ν-1)) with Kν(-ν-1), Kνn(ν) [swap]
#    ΔG_mono = G_ν - G_{-ν-1}
# ============================================================

# ── helpers ──────────────────────────────────────────────────

"""
ω, ν (primary), ν_neg (secondary=-ν-1) を指定して G = Bref/(2iωBinc) を計算。
Kν は ν + fn_p 由来、Kνn は ν_neg + fn_m 由来。
"""
function compute_G_branch(p, ν, fn_p, ν_neg, fn_m; nmax=40)
    s, ϵ, κ = p.s, p.ϵ, p.κ
    ω_c = p.ω

    Ap  = BHPtoolkit.compute_Aplus(p, ν, fn_p; nmax=nmax)
    Am  = BHPtoolkit.compute_Aminus(p, ν, fn_p; nmax=nmax)
    Kν  = BHPtoolkit.compute_Knu(p, ν,     fn_p; nmax=nmax)
    Kνn = BHPtoolkit.compute_Knu(p, ν_neg, fn_m; nmax=nmax)

    phase      = exp(-im * (ϵ * log(ϵ) - (1 - κ) / 2 * ϵ))
    phase_conj = exp( im * (ϵ * log(ϵ) - (1 - κ) / 2 * ϵ))
    sinν_factor = sin(π * (ν - s + im*ϵ)) / sin(π * (ν + s - im*ϵ))

    Binc = ω_c^(-1) * (Kν - im * exp(-im*π*ν) * sinν_factor * Kνn) * Ap * phase
    Bref = ω_c^(-1 - 2s) * (Kν + im * exp(im*π*ν) * Kνn) * Am * phase_conj

    return Bref / (2im * ω_c * Binc)
end

# ── 計算 ─────────────────────────────────────────────────────

s, l, m = -2, 2, 2
a       = 0.9
nmax    = 40
nmax_cf = 150

# BigFloat 設定（法1の参照値用）
setprecision(BigFloat, 256)
a_bf = BigFloat(string(a))
δ_bf = BigFloat("1e-6")

# ─────────────────────────────────────────────────────────────
# δ→0 の極限を正しく取ると：
#
#   G_R → H_m(-iσ)          / (2σ)   where H = Bref/Binc
#   G_L → conj(H_{-m}(+iσ)) / (2σ)
#
#   ΔG  = [H_m(-iσ) - conj(H_{-m}(+iσ))] / (2σ)
#
# 2点での評価だが δオフセット不要 → Float64 で計算可能なはず
# ─────────────────────────────────────────────────────────────

println("σ\t\t|ΔG_BF(256bit)|\t|ΔG_F64直接|\t|ΔG_mono_誤|")
println("(ref BF)\t(δ=1e-6あり)\t(δなし)\t\t(ν,-ν-1 swap)")
println("-"^80)

for σ_val in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    # ── 法1: BigFloat 引き算（参照値） ──────────────────────
    σ_bf = BigFloat(string(σ_val))
    ω_R   = δ_bf - im * σ_bf
    amp_R = compute_amplitudes(s, l, m, a_bf, ω_R; nmax=nmax, nmax_cf=nmax_cf)
    G_R   = amp_R.Bref / (2im * ω_R * amp_R.Binc)
    ω_mir = δ_bf + im * σ_bf
    amp_mir = compute_amplitudes(s, l, -m, a_bf, ω_mir; nmax=nmax, nmax_cf=nmax_cf)
    G_L   = conj(amp_mir.Bref) / (2im * (-δ_bf - im*σ_bf) * conj(amp_mir.Binc))
    ΔG_bf = ComplexF64(G_R - G_L)

    # ── 法2: Float64、δオフセットなし ───────────────────────
    # H_m(-iσ) と conj(H_{-m}(+iσ)) の差 / (2σ)
    ω_neg  = -im * σ_val          # 負虚軸
    ω_pos  = +im * σ_val          # 正虚軸
    amp_neg = compute_amplitudes(s, l,  m, a, ω_neg; nmax=nmax, nmax_cf=nmax_cf)
    amp_pos = compute_amplitudes(s, l, -m, a, ω_pos; nmax=nmax, nmax_cf=nmax_cf)
    H_neg  = amp_neg.Bref / amp_neg.Binc    # Bref_m(-iσ)/Binc_m(-iσ)
    H_pos  = amp_pos.Bref / amp_pos.Binc    # Bref_{-m}(+iσ)/Binc_{-m}(+iσ)
    ΔG_f64 = (H_neg - conj(H_pos)) / (2σ_val)

    # ── 法3: ν,-ν-1 swap（前回の誤った試み） ───────────────
    ν, p   = compute_nu(s, l, m, a, ω_neg; nmax_cf=nmax_cf)
    ν_neg  = -ν - 1
    fn_p   = compute_fn(p, ν;     nmax=nmax)
    fn_m_  = compute_fn(p, ν_neg; nmax=nmax)
    G_ν    = compute_G_branch(p, ν,     fn_p, ν_neg, fn_m_; nmax=nmax)
    G_νn   = compute_G_branch(p, ν_neg, fn_m_, ν,    fn_p;  nmax=nmax)
    ΔG_mono = G_ν - G_νn

    @printf("%.1f\t\t%.4e\t%.4e\t%.4e\n",
            σ_val, abs(ΔG_bf), abs(ΔG_f64), abs(ΔG_mono))
end

println()
println("法1 (BF ref): BigFloat 256bit, δ=1e-6 オフセットあり")
println("法2 (F64直接): Float64, ω=±iσ 評価 → δオフセット不要")
println("法3 (mono誤): ν と -ν-1 swap → G が対称なので常に≈0")
