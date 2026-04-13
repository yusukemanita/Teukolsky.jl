using Pkg; Pkg.activate("/Users/yusuke/work/BHPtoolkit.jl")
using BHPtoolkit, Printf

s,l,m = -2,2,2; a=0.0; r=10.0

println("ω         Re(ν)    Im(ν)    |Rin_raw|    |Rin_phys|     case")
println("─"^72)

ν_prev = nothing
for ω in range(0.05, 1.0; length=50)
    p    = MSTParams(s,l,m,a,ω)
    ν, _ = compute_nu(s,l,m,a,ω; ν_init=ν_prev)
    c2pn = BHPtoolkit.monodromy_cos2pi_nu(s,l,m,a,ω,p.λ)
    rc   = real(c2pn)
    case = rc < -1 ? "half" : (rc > 1 ? "int " : "real")
    fn_d = compute_fn(p, ν; nmax=40)
    Rv      = Rin(p, ν, fn_d, r)
    Rv_phys = Rin_phys(p, ν, fn_d, r)
    @printf "%.3f  %8.4f  %8.4f  %.4e   %.4e   %s\n" ω real(ν) imag(ν) abs(Rv) abs(Rv_phys) case
    global ν_prev = ν
end

println()
println("Rin_raw  : raw MST series (normalization depends on ν branch → jumps at transitions)")
println("Rin_phys : = Rin_raw / Btrans  (smooth; matches Teukolsky package convention)")
println("Waveform : G = Rin_raw × Bref / (2iω × Binc) is also smooth (Btrans cancels)")
