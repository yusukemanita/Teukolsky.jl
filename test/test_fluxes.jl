using Test
using BHPtoolkit

# ============================================================
#  B5 — point-particle source convolution + fluxes
#  s=-2 circular equatorial. Validate Z^∞/Z^H and the energy / angular-momentum
#  fluxes against Wolfram TeukolskyPointParticleMode (test/flux_ref.txt).
# ============================================================

# Parse a Mathematica scalar token: strip precision backtick, *^→e.
_pnum(t) = parse(Float64, replace(split(t, "`")[1], "*^" => "e"))

# Parse flux_ref.txt into a vector of case Dicts.
function _read_flux_cases(path)
    cases = Dict{String,Any}[]; cur = nothing
    for ln in readlines(path)
        st = strip(ln)
        (isempty(st) || startswith(st, "#")) && continue
        if startswith(st, "CASE")
            cur === nothing || push!(cases, cur)
            f = split(st)
            g(k) = _pnum(split(f[findfirst(x -> startswith(x, k), f)], "=")[2])
            cur = Dict{String,Any}("s"=>Int(g("s")), "l"=>Int(g("l")),
                                   "m"=>Int(g("m")), "a"=>g("a"), "p"=>g("p"))
        else
            f = split(st)
            # drop Mathematica "+" / "...I" tokens (complex written "Re + Im*I")
            vals = [_pnum(t) for t in f[2:end] if t != "+" && !occursin("I", t)]
            cur[f[1]] = length(vals) == 1 ? vals[1] : complex(vals[1], vals[2])
        end
    end
    cur === nothing || push!(cases, cur)
    return cases
end

@testset "B5 point-particle fluxes (s=-2 circular)" begin
    for c in _read_flux_cases(joinpath(@__DIR__, "flux_ref.txt"))
        s, l, m, a, p = c["s"], c["l"], c["m"], c["a"], c["p"]
        @testset "s=$s l=$l m=$m a=$a p=$p" begin
            # a<0 with x=+1 is the retrograde convention used by the reference.
            md = TeukolskyPointParticleMode(s, l, m, a, p; prograde=true)

            @test isapprox(md.ω, c["omega"]; rtol=1e-10)
            # asymptotic amplitudes (phase-invariant: compare magnitudes)
            @test isapprox(abs(md.Z.ZInf), abs(c["ZInf"]); rtol=1e-7)
            @test isapprox(abs(md.Z.ZHor), abs(c["ZHor"]); rtol=5e-3)
            # energy fluxes
            @test isapprox(md.EnergyFlux.Inf, c["EnergyFluxInf"]; rtol=1e-7)
            @test isapprox(md.EnergyFlux.Hor, c["EnergyFluxHor"]; rtol=5e-3)
            # angular-momentum flux = (m/ω)·energy flux, exactly
            @test md.AngularMomentumFlux.Inf ≈ md.EnergyFlux.Inf * m / real(md.ω)
            @test isapprox(md.AngularMomentumFlux.Inf, c["AngMomFluxInf"]; rtol=1e-7)
        end
    end

    @testset "physics sanity + guards" begin
        # l=m=2, a=0, p=10 energy flux to infinity (Schwarzschild benchmark)
        md0 = TeukolskyPointParticleMode(-2, 2, 2, 0.0, 10.0)
        @test isapprox(md0.EnergyFlux.Inf, 2.684397739103742e-5; rtol=1e-6)
        # Kerr prograde horizon flux is NEGATIVE (superradiance, m>0)
        mk = TeukolskyPointParticleMode(-2, 2, 2, 0.9, 6.0)
        @test mk.EnergyFlux.Hor < 0
        # only s=-2 circular supported
        @test_throws ArgumentError TeukolskyPointParticleMode(0, 2, 2, 0.0, 10.0)
        @test_throws ArgumentError KerrCircularOrbit(1.2, 10.0)   # |a|>1
    end
end
