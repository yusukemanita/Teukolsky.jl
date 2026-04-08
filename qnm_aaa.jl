using Pkg
Pkg.activate("/Users/yusuke/work/BHPtoolkit.jl")

using BHPtoolkit
using BaryRational
using Plots
using LinearAlgebra
using Printf
using LaTeXStrings

# ── Physical parameters ──
s, l, m, a = -2, 2, 2, 0.0

omega = filter(!iszero, -3.0:0.01:3.0)
Gapp(ω) =  begin
        if real(ω) > 0
            amp = compute_amplitudes(s, l, m, a, ω)
            Binc = amp.Binc
            Bref = amp.Bref

            G = Bref / (2im * ω * Binc)
        else
            amp = compute_amplitudes(s, l, -m, a, -conj(ω))
            Binc = conj(amp.Binc)
            Bref = conj(amp.Bref)

            G = Bref / (2im * ω * Binc)
        end
        return G
end

omega_strip = ComplexF64.(omega) .- 0.1im
GF = [Gapp(ω) for ω in omega_strip]

using CSV, DataFrames
# QNMデータを読み込む関数
function load_qnm_data(l::Int, m::Int, n::Int, a_target::Float64)
    filepath = "/Users/yusuke/Downloads/KerrQNMEFs-2/l2/s-2l$(l)m$(m)n$n.dat"
    df = CSV.read(filepath, DataFrame, header=false)
    a = df[!, 1]
    idx = argmin(abs.(a .- a_target))
    ω = (df[idx, 2] + im * df[idx, 3]) / 2
    B = (df[idx, 8] + im * df[idx, 9]) / 16
    return ω, B
end

# Load known QNM frequencies and amplitudes for (l=2, m=±2) from CSV files
ω22 = Vector{ComplexF64}(undef, 8)
B22 = Vector{ComplexF64}(undef, 8)

for n in 0:7
    ω22[n+1], B22[n+1] = load_qnm_data(l, m, n, a)
end

ω22R = Vector{ComplexF64}(undef, 8)
B22R = Vector{ComplexF64}(undef, 8)

for n in 0:7
    ω22R[n+1], B22R[n+1] = load_qnm_data(l, -m, n, a)
    ω22R[n+1] = -conj(ω22R[n+1])
    B22R[n+1] = conj(B22R[n+1])
end

ω_known = [ω22; ω22R]

# AAA approximation
approx_freq = aaa(omega_strip, GF; tol=1e-13, mmax=150)
psi4_aaa = approx_freq.(omega_strip)
println("Support points: $(length(approx_freq.x))")
poles_freq, res_freq, zeros_freq = prz(approx_freq)

# ── Plot 1: ψ̃₄(ω) 元データ vs AAA近似 ──────────────────────────
p1 = plot(real.(omega_strip), abs.(GF),
    label="Original (Im=-0.8i)", lw=1, alpha=0.7,
    xlabel=L"\Re(\omega)", ylabel=L"|\tilde{\psi}_4(\omega - 0.8i)|",
    yscale=:log10, framestyle=:box, grid=true,
    fontfamily="Computer Modern", dpi=100)
plot!(p1, real.(omega_strip), abs.(psi4_aaa),
    label="AAA", ls=:dash, lw=1.5)

# ── Plot 2: 複素ω平面上のポールとQNM参照値 ─────────────────────
res_log = log10.(clamp.(abs.(res_freq), 1e-30, Inf))
p2 = scatter(real.(ω_known), imag.(ω_known),
    label="QNM reference", marker=:circle, ms=8, alpha=0.7, color=:blue)
scatter!(p2, real.(poles_freq), imag.(poles_freq),
    label="AAA poles", marker=:circle, ms=5,
    marker_z=res_log, color=:plasma, colorbar=true, colorbar_title="log|Res|",
    xlabel=L"\Re(\omega)", ylabel=L"\Im(\omega)",
    ylim=(-2.0, 2.0),
    framestyle=:box, grid=true,
    fontfamily="Computer Modern", dpi=100)
scatter!(p2, real.(zeros_freq), imag.(zeros_freq),
    label="AAA zeros", marker=:circle, ms=4, color=:green)

plot(p1, p2, layout=(2,1), size=(800, 700))