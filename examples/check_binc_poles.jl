using Pkg
Pkg.activate("/Users/yusuke/work/BHPtoolkit.jl")

using BHPtoolkit
using CSV, DataFrames
using Printf

s, l, m, a = -2, 2, 2, 0.9

function load_qnm(l, m, n, a_target)
    fp = "/Users/yusuke/Downloads/KerrQNMEFs-2/l2/s-2l$(l)m$(m)n$n.dat"
    df = CSV.read(fp, DataFrame, header=false)
    idx = argmin(abs.(df[!, 1] .- a_target))
    return (df[idx, 2] + im * df[idx, 3]) / 2
end

# Newton refinement: find ω where Binc(ω) = 0
function refine_qnm(ω0; prec=128, maxiter=30, tol=1e-20)
    ω = Complex{BigFloat}(ω0)
    δ = BigFloat("1e-6")
    for _ in 1:maxiter
        setprecision(BigFloat, prec) do
            f  = compute_amplitudes(s, l, m, a, ω; precision=prec).Binc
            fp = (compute_amplitudes(s, l, m, a, ω+δ; precision=prec).Binc -
                  compute_amplitudes(s, l, m, a, ω-δ; precision=prec).Binc) / (2δ)
            Δ  = -f / fp
            ω += Δ
            abs(Float64(abs(Δ))) < tol && return
        end
    end
    return ω
end

println("Checking |Binc| at reference QNM frequencies and refining:")
println("-"^90)

for n in 0:7
    ω_ref = load_qnm(l, m, n, a)

    Binc_f64  = compute_amplitudes(s, l, m, a, ω_ref).Binc
    Binc_128  = setprecision(BigFloat,128) do
        compute_amplitudes(s, l, m, a, Complex{BigFloat}(ω_ref); precision=128).Binc
    end

    @printf("n=%d  ω_ref=(%+.5f%+.5fim)  |B|_F64=%.2e  |B|_128=%.2e\n",
            n, real(ω_ref), imag(ω_ref),
            abs(Binc_f64), Float64(abs(Binc_128)))
end

println("\nNewton-refined QNM frequencies (128-bit):")
println("-"^90)
for n in 0:7
    ω_ref = load_qnm(l, m, n, a)
    try
        ω_refined = refine_qnm(ω_ref)
        Binc_check = setprecision(BigFloat,128) do
            compute_amplitudes(s, l, m, a, ω_refined; precision=128).Binc
        end
        @printf("n=%d  ω_ref=(%+.6f%+.6fim)  →  ω_refined=(%+.6f%+.6fim)  |Binc|=%.2e\n",
                n, real(ω_ref), imag(ω_ref),
                Float64(real(ω_refined)), Float64(imag(ω_refined)),
                Float64(abs(Binc_check)))
    catch e
        @printf("n=%d  failed: %s\n", n, e)
    end
end
