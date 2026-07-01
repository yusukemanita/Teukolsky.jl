using Test
using Teukolsky

# ============================================================
#  Shared elliptic-integral / Jacobi module (src/elliptic_integrals.jl)
#
#  Cross-check against the Wolfram reference (test/elliptic_ref.txt), all in
#  the PARAMETER m = k² convention used by Mathematica and KerrGeodesics:
#     KE ; m ; EllipticK[m] ; EllipticE[m]
#     F  ; phi ; m ; EllipticF[phi,m]
#     EI ; phi ; m ; EllipticE[phi,m]
#     PIC; n ; m ; EllipticPi[n,m]
#     PII; n ; phi ; m ; EllipticPi[n,phi,m]
#     SN ; u ; m ; JacobiSN[u,m]
#     AM ; u ; m ; JacobiAmplitude[u,m]
# ============================================================

# parse "<x>e<y>" InputForm number to type T
_pe(::Type{T}, s) where {T} = parse(T, strip(s))

@testset "Elliptic integrals & Jacobi functions" begin

    refrows = [split(strip(ln), ";") for ln in
               readlines(joinpath(@__DIR__, "elliptic_ref.txt")) if !isempty(strip(ln))]

    @testset "Carlson symmetric forms (analytic identities)" begin
        # R_F(x,x,x) = x^{-1/2}; R_C(x,x)=x^{-1/2}; R_D(x,x,x)=x^{-3/2}
        for x in (0.3, 1.0, 2.7, 13.0)
            @test isapprox(rf(x, x, x), 1 / sqrt(x); rtol=1e-14)
            @test isapprox(rc(x, x), 1 / sqrt(x); rtol=1e-14)
            @test isapprox(rd(x, x, x), x^(-1.5); rtol=1e-14)
            @test isapprox(rj(x, x, x, x), x^(-1.5); rtol=1e-14)
        end
        # R_C(0,y) = (π/2)/√y
        @test isapprox(rc(0.0, 2.0), (π/2)/sqrt(2.0); rtol=1e-14)
        # R_D(0,y,y) = (3π/4) y^{-3/2}
        @test isapprox(rd(0.0, 1.7, 1.7), (3π/4)*1.7^(-1.5); rtol=1e-13)
        # R_J reduces to R_D when p=z
        @test isapprox(rj(0.4, 1.1, 2.2, 2.2), rd(0.4, 1.1, 2.2); rtol=1e-13)
        # symmetry of R_F
        @test isapprox(rf(1.0, 2.0, 3.0), rf(3.0, 1.0, 2.0); rtol=1e-14)
        # R_J with negative p (principal value): finite & symmetric in x,y,z
        @test isfinite(rj(0.5, 1.0, 2.0, -0.7))
        @test isapprox(rj(0.5, 1.0, 2.0, -0.7), rj(2.0, 0.5, 1.0, -0.7); rtol=1e-12)
    end

    # ---- Float64 vs Wolfram ----
    maxrel = 0.0
    @testset "Float64 vs Wolfram reference" begin
        for f in refrows
            tag = f[1]
            if tag == "KE"
                m = _pe(Float64, f[2])
                for (got, ref) in ((ellK(m), _pe(Float64, f[3])),
                                   (ellE(m), _pe(Float64, f[4])))
                    rel = abs(got - ref) / abs(ref)
                    maxrel = max(maxrel, rel)
                    @test rel < 1e-13
                end
            elseif tag == "F"
                φ = _pe(Float64, f[2]); m = _pe(Float64, f[3]); ref = _pe(Float64, f[4])
                rel = abs(ellF(φ, m) - ref) / abs(ref)
                maxrel = max(maxrel, rel); @test rel < 1e-13
            elseif tag == "EI"
                φ = _pe(Float64, f[2]); m = _pe(Float64, f[3]); ref = _pe(Float64, f[4])
                rel = abs(ellEinc(φ, m) - ref) / abs(ref)
                maxrel = max(maxrel, rel); @test rel < 1e-13
            elseif tag == "PIC"
                n = _pe(Float64, f[2]); m = _pe(Float64, f[3]); ref = _pe(Float64, f[4])
                rel = abs(ellPi(n, m) - ref) / abs(ref)
                maxrel = max(maxrel, rel); @test rel < 1e-13
            elseif tag == "PII"
                n = _pe(Float64, f[2]); φ = _pe(Float64, f[3])
                m = _pe(Float64, f[4]); ref = _pe(Float64, f[5])
                rel = abs(ellPi(n, φ, m) - ref) / abs(ref)
                maxrel = max(maxrel, rel); @test rel < 1e-13
            elseif tag == "SN"
                u = _pe(Float64, f[2]); m = _pe(Float64, f[3]); ref = _pe(Float64, f[4])
                rel = abs(jacobi_sn(u, m) - ref) / (abs(ref) + 1e-300)
                maxrel = max(maxrel, rel); @test rel < 1e-12
            elseif tag == "AM"
                u = _pe(Float64, f[2]); m = _pe(Float64, f[3]); ref = _pe(Float64, f[4])
                rel = abs(jacobi_am(u, m) - ref) / abs(ref)
                maxrel = max(maxrel, rel); @test rel < 1e-12
            end
        end
        @info "Elliptic Float64 max relative error vs Wolfram" maxrel
    end

    # ---- BigFloat (~1e-25) ----
    @testset "BigFloat vs Wolfram reference (~1e-25)" begin
        setprecision(BigFloat, 113) do   # ≈ 34 decimal digits
            bigmax = big"0.0"
            for f in refrows
                tag = f[1]
                if tag == "KE"
                    m = _pe(BigFloat, f[2])
                    for (got, ref) in ((ellK(m), _pe(BigFloat, f[3])),
                                       (ellE(m), _pe(BigFloat, f[4])))
                        bigmax = max(bigmax, abs(got - ref) / abs(ref))
                    end
                elseif tag == "F"
                    got = ellF(_pe(BigFloat, f[2]), _pe(BigFloat, f[3]))
                    bigmax = max(bigmax, abs(got - _pe(BigFloat, f[4])) / abs(_pe(BigFloat, f[4])))
                elseif tag == "EI"
                    got = ellEinc(_pe(BigFloat, f[2]), _pe(BigFloat, f[3]))
                    bigmax = max(bigmax, abs(got - _pe(BigFloat, f[4])) / abs(_pe(BigFloat, f[4])))
                elseif tag == "PIC"
                    got = ellPi(_pe(BigFloat, f[2]), _pe(BigFloat, f[3]))
                    bigmax = max(bigmax, abs(got - _pe(BigFloat, f[4])) / abs(_pe(BigFloat, f[4])))
                elseif tag == "PII"
                    got = ellPi(_pe(BigFloat, f[2]), _pe(BigFloat, f[3]), _pe(BigFloat, f[4]))
                    bigmax = max(bigmax, abs(got - _pe(BigFloat, f[5])) / abs(_pe(BigFloat, f[5])))
                elseif tag == "SN"
                    got = jacobi_sn(_pe(BigFloat, f[2]), _pe(BigFloat, f[3]))
                    bigmax = max(bigmax, abs(got - _pe(BigFloat, f[4])) / (abs(_pe(BigFloat, f[4])) + big"1e-300"))
                elseif tag == "AM"
                    got = jacobi_am(_pe(BigFloat, f[2]), _pe(BigFloat, f[3]))
                    bigmax = max(bigmax, abs(got - _pe(BigFloat, f[4])) / abs(_pe(BigFloat, f[4])))
                end
            end
            @info "Elliptic BigFloat(113) max relative error vs Wolfram" bigmax
            @test bigmax < 1e-25
        end
    end

    # ---- type genericity / consistency ----
    @testset "Type genericity & internal consistency" begin
        # m=0 limits
        @test isapprox(ellK(0.0), π/2; rtol=1e-15)
        @test isapprox(jacobi_sn(0.7, 0.0), sin(0.7); rtol=1e-14)
        @test isapprox(jacobi_am(0.7, 0.0), 0.7; rtol=1e-14)
        # sn = sin(am)
        @test isapprox(jacobi_sn(2.3, 0.4), sin(jacobi_am(2.3, 0.4)); rtol=1e-14)
        # complete = incomplete at φ=π/2
        @test isapprox(ellF(π/2, 0.6), ellK(0.6); rtol=1e-14)
        @test isapprox(ellEinc(π/2, 0.6), ellE(0.6); rtol=1e-14)
        @test isapprox(ellPi(0.3, π/2, 0.6), ellPi(0.3, 0.6); rtol=1e-13)
        # BigFloat return types
        @test ellK(big"0.5") isa BigFloat
        @test jacobi_sn(big"1.0", big"0.5") isa BigFloat
    end
end
