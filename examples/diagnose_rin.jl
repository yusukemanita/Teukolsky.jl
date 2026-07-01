using Pkg; Pkg.activate("/Users/yusuke/work/Teukolsky.jl")
using Teukolsky, Printf

s,l,m = -2,2,2; a=0.0; r=10.0

println("ω         Re(ν)    Im(ν)    |Rin_raw|    |Rin|           case")
println("─"^72)

ν_prev = nothing
for ω in range(0.05, 1.0; length=50)
    p    = MSTParams(s,l,m,a,ω)
    ν, _ = compute_nu(s,l,m,a,ω; ν_init=ν_prev)
    c2pn = Teukolsky.monodromy_cos2pi_nu(s,l,m,a,ω,p.λ)
    rc   = real(c2pn)
    case = rc < -1 ? "half" : (rc > 1 ? "int " : "real")
    fn_d = compute_fn(p, ν; nmax=40)
    Rv_raw  = Teukolsky._Rin_raw(p, ν, fn_d, r)
    Rv      = Rin(p, ν, fn_d, r)
    @printf "%.3f  %8.4f  %8.4f  %.4e   %.4e   %s\n" ω real(ν) imag(ν) abs(Rv_raw) abs(Rv) case
    global ν_prev = ν
end

println()
println("Rin_raw  : raw MST series (internal; normalization depends on ν branch → jumps at transitions)")
println("Rin      : = Rin_raw / Btrans  (smooth; matches Teukolsky package convention)")
