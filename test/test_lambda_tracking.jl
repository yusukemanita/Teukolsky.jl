# Spheroidal eigenvalue branch tracking (λ continuity) regression.
#
# The spectral-matrix branch used to be selected as the eigenvalue CLOSEST to
# the O(c²) perturbative λ — a criterion that degenerates far from c = 0: at
# a=0.7, l=m=2, s=±2, ω=iσ near σ ≈ 4.1885 (|c| ≈ 2.93) two well-separated
# eigenvalues (Δλ ≈ 1.16) sit at |λ−λ_pert| = 2.802 vs 2.810, the argmin flips,
# and λ (hence ν, by ≈0.06) jumped discontinuously across the sweep.
#
# _swsh_eigen now selects by analytic continuation from c = 0 along the ray t·c,
# matching eigenvectors by overlap — the defining label of the spheroidal
# harmonic (the branch continuously connected to ₛY_lm).  These tests pin:
#   (a) continuity across the former jump,
#   (b) global smoothness of λ(σ) on PIA sweeps for several (s,l,m,a),
#   (c) agreement with a fine-step (400) continuation arbiter at hard points,
#   (d) the c → 0 limit and a small-c perturbative cross-check.
using Test
using LinearAlgebra
using Teukolsky

@testset "spheroidal λ branch tracking" begin
    @testset "former jump at σ≈4.1885 (a=0.7, s=-2, l=m=2)" begin
        λ1 = ComplexF64(Teukolsky.compute_lambda(-2, 2, 2, 0.7, 4.188im))
        λ2 = ComplexF64(Teukolsky.compute_lambda(-2, 2, 2, 0.7, 4.189im))
        @test abs(λ2 - λ1) < 0.01           # was ≈ 1.16
    end

    @testset "PIA sweep smoothness" begin
        for (s,l,m,a) in ((-2,2,2,0.7), (2,2,2,0.7), (-2,3,2,0.7), (-2,2,2,0.99))
            σs = 0.2:0.05:8.0
            λs = [ComplexF64(Teukolsky.compute_lambda(s,l,m,a,im*σ)) for σ in σs]
            dλ = abs.(diff(λs))
            med = sort(dλ)[div(end,2)]
            # smooth curve: no step exceeds a small multiple of the median slope
            @test maximum(dλ) < 10*med + 0.05
        end
    end

    @testset "fine-step continuation arbiter" begin
        function lambda_arbiter(s,l,m,a,ω; nstep=400, l_max=20)
            c = ComplexF64(a*ω); lmin=max(abs(m),abs(s)); ells=lmin:l_max
            N=length(ells); il=l-lmin+1
            v=zeros(ComplexF64,N); v[il]=1
            local F, k
            for j in 1:nstep
                t = j/nstep
                Mt = [Teukolsky.M_matrix_elem(s,t*c,m,li,lj) for li in ells, lj in ells]
                F  = eigen(Mt)
                k  = argmax([abs(dot(v, view(F.vectors,:,jj))) for jj in 1:N])
                v  = F.vectors[:,k]
            end
            return F.values[k] - 2m*c + c^2
        end
        for (s,l,m,a,σ) in ((-2,2,2,0.7,4.2), (-2,2,2,0.99,8.3),
                            (-2,3,2,0.7,6.1), (-2,4,3,0.9,9.9))
            λp = ComplexF64(Teukolsky.compute_lambda(s,l,m,a,im*σ))
            λa = lambda_arbiter(s,l,m,a,im*σ)
            @test abs(λp - λa) / abs(λa) < 1e-12
        end
    end

    @testset "c→0 limit and small-c cross-check" begin
        # exact spherical limit
        @test abs(ComplexF64(Teukolsky.compute_lambda(-2,2,2,0.0,0.5)) -
                  (2*3 - (-2)*(-1))) < 1e-13
        # small-c perturbative agreement to O(c³)
        s,l,m,a,ω = -2,2,2,0.7,0.01
        c = a*ω
        λ0 = l*(l+1)-s*(s+1); λ1 = -2m*(1+s^2/(l*(l+1)))
        H(ℓ)= 2*(ℓ^2-m^2)*(ℓ^2-s^2)/((2ℓ-1)*ℓ^3*(2ℓ+1))
        λpert = λ0 + c*λ1 + c^2*(H(l+1)-H(l))
        @test abs(ComplexF64(Teukolsky.compute_lambda(s,l,m,a,ω)) - λpert) < 1e-4
    end
end
