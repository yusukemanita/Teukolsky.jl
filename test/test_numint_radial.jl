using Test
using BHPtoolkit

# ============================================================
#  B3 — NumericalIntegration radial backend
#
#  Cross-check the direct-ODE backend against (a) the Wolfram
#  TeukolskyRadial NumericalIntegration reference (test/numint_ref.txt:
#  s;l;m;a;ω;bc(0=In,1=Up);r;ReR;ImR;ReR';ImR') and (b) the MST Rin/Rup where
#  MST is accurate. Also verify the MST solution satisfies the radial ODE
#  (reference-free).
# ============================================================

@testset "B3 NumericalIntegration radial backend" begin
    # group the reference rows by (s,l,m,a,ω)
    rows = Dict{NTuple{5,Any}, Vector{NTuple{6,Float64}}}()
    for ln in readlines(joinpath(@__DIR__, "numint_ref.txt"))
        isempty(strip(ln)) && continue
        f = split(strip(ln), ";")
        key = (parse(Int,f[1]), parse(Int,f[2]), parse(Int,f[3]),
               parse(Float64,f[4]), parse(Float64,f[5]))
        push!(get!(rows, key, NTuple{6,Float64}[]),
              (parse(Float64,f[6]), parse(Float64,f[7]),
               parse(Float64,f[8]), parse(Float64,f[9]),
               parse(Float64,f[10]), parse(Float64,f[11])))
    end

    @testset "vs Wolfram numint + MST  s=$s l=$l m=$m a=$a ω=$ω" for ((s,l,m,a,ω), data) in rows
        ni = NumericalIntegrationRadial(s, l, m, a, ω)
        ν, p = compute_nu(s, l, m, a, ω); fn = compute_fn(p, ν)
        for (bc, r, ReR, ImR, ReRp, ImRp) in data
            Rref = complex(ReR, ImR)
            if bc == 0   # In: integrated outward from the horizon — high accuracy
                @test isapprox(ni.In(r), Rref; rtol=1e-9)
                @test isapprox(ni.In(r; deriv=1), complex(ReRp, ImRp); rtol=1e-8)
                r ≤ 12 && @test isapprox(ni.In(r), Rin(p, ν, fn, r); rtol=1e-9)  # MST accurate at small r
            else         # Up: integrated inward — looser (in-mode contamination)
                @test isapprox(ni.Up(r), Rref; rtol=1e-5)
            end
        end
    end

    @testset "MST solution satisfies the radial ODE (reference-free)" begin
        for (s,l,m,a,ω) in [(-2,2,2,0.0,0.5), (-2,2,2,0.9,0.5), (0,2,0,0.7,0.3)]
            ν, p = compute_nu(s, l, m, a, ω); fn = compute_fn(p, ν)
            for r in (4.0, 7.0)
                h = 1e-5
                Rpp_fd = (dRin(p,ν,fn,r+h) - dRin(p,ν,fn,r-h)) / (2h)
                _, Rpp_ode = BHPtoolkit._teuk_radial_rhs(r, Rin(p,ν,fn,r), dRin(p,ν,fn,r),
                                                         s, m, a, complex(ω), p.λ)
                @test isapprox(Rpp_fd, Rpp_ode; rtol=1e-5)   # FD-limited
            end
        end
    end

    @testset "BigFloat backend" begin
        ni = setprecision(BigFloat, 128) do
            NumericalIntegrationRadial(-2, 2, 2, big"0.0", Complex{BigFloat}(big"0.5"))
        end
        v = setprecision(() -> ni.In(BigFloat(6)), BigFloat, 128)
        @test v isa Complex{BigFloat}
        # matches MST at a radius where MST is accurate, to well beyond Float64
        vmst = setprecision(BigFloat, 128) do
            ν, p = compute_nu(-2, 2, 2, big"0.0", Complex{BigFloat}(big"0.5"))
            Rin(p, ν, compute_fn(p, ν), BigFloat(6))
        end
        @test isapprox(v, vmst; rtol=1e-18)
    end
end
