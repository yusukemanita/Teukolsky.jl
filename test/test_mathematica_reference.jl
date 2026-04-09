using Test
using BHPtoolkit

# ============================================================
#  Mathematica BHPToolkit (Teukolsky-1.1.1) との比較テスト
#
#  参照値は generate_reference_mathematica.wls で生成。
#  Method -> "MST", WorkingPrecision -> 32 で計算。
#  λ = SpinWeightedSpheroidalEigenvalue (MST/Sasaki-Tagoshi 規約)
# ============================================================

# Mathematica の参照値（精度注釈を除去して Julia 浮動小数点に変換）
const ref = (;
    # s=-2, l=2, m=2, a=0, ω=0.3
    sch_om03 = (;
        s=(-2), l=2, m=2, a=0.0, ω=0.3,
        λ    =  4.0 + 0.0im,
        ν    =  1.7792805424199195 + 0.0im,
        Binc =  337.6716131917701  + 74.91861048476544im,
        Bref = -1.7553180132857615 - 0.3790959765590263im,
    ),
    # s=-2, l=2, m=2, a=0, ω=0.5
    sch_om05 = (;
        s=(-2), l=2, m=2, a=0.0, ω=0.5,
        λ    =  4.0 + 0.0im,
        ν    =  0.5 + 0.36188061539416753im,
        Binc = -41.18358012424982  - 30.069530115741246im,
        Bref = -0.2551027281545793 + 0.026707788816647016im,
    ),
    # s=-2, l=2, m=2, a=0, ω=1.0  (ν は枝分かれあり: λ のみ比較)
    sch_om10 = (;
        s=(-2), l=2, m=2, a=0.0, ω=1.0,
        λ    =  4.0 + 0.0im,
        Binc = -10.471758315257257 + 35.36017936904070im,
        Bref = -0.0014736564497268391 + 0.0006299694738176936im,
    ),
    # s=-2, l=2, m=2, a=0.5, ω=0.3
    ker05_om03 = (;
        s=(-2), l=2, m=2, a=0.5, ω=0.3,
        λ    = 3.005755687507320  + 0.0im,
        ν    = 1.8092270433414369 + 0.0im,
        Binc = 158.33672472263933 + 326.94074750465346im,
        Bref = -1.4999793026087090 - 1.8233938259556797im,
    ),
    # s=-2, l=2, m=2, a=0.9, ω=0.3
    ker09_om03 = (;
        s=(-2), l=2, m=2, a=0.9, ω=0.3,
        λ    = 2.2181476863133891  + 0.0im,
        ν    = 1.8187092400025136  + 0.0im,
        Binc = 156.58379703808201  + 445.08445202688828im,
        Bref = -2.3604659264561087 - 2.7204852443190117im,
    ),
    # s=-2, l=2, m=2, a=0.9, ω=0.5
    ker09_om05 = (;
        s=(-2), l=2, m=2, a=0.9, ω=0.5,
        λ    = 1.0483733822648650  + 0.0im,
        ν    = 0.5 + 0.49070747023206875im,
        Binc = 9.3650561467627418  + 1.8746098582284640im,
        Bref = -0.02750610557358242 - 0.6829848768952166im,
    ),
    # s=0, l=2, m=0, a=0, ω=0.3
    sch_s0_om03 = (;
        s=0, l=2, m=0, a=0.0, ω=0.3,
        λ    =  6.0 + 0.0im,
        ν    =  1.8522206563655083 + 0.0im,
        Binc =  46.65012132174134  - 22.55216822137581im,
        Bref = -10.65395810599658  + 50.66879994132571im,
    ),
)

# ── テスト ────────────────────────────────────────────────────
# 注意: Binc, Bref の絶対値はコード間の規約で異なるが、
#       物理的に意味のある比率 Bref/Binc は一致する。
#       G(ω) = Bref / (2iω Binc) はこの比率で決まる。
tol_λ   = 1e-10  # λ の許容相対誤差
tol_rat = 1e-8   # Bref/Binc の許容相対誤差

@testset "vs Mathematica BHPToolkit (Teukolsky-1.1.1)" begin

    for (key, r) in pairs(ref)
        has_nu = hasproperty(r, :ν)

        @testset "$(key)" begin
            s, l, m, a, ω = r.s, r.l, r.m, r.a, r.ω

            ν, p   = compute_nu(s, l, m, a, ω)
            amp    = compute_amplitudes(s, l, m, a, ω)

            # λ (angular eigenvalue)
            @test abs(p.λ - r.λ) / abs(r.λ) < tol_λ

            # ν (renormalized angular momentum) — 実数の場合のみ
            if has_nu && abs(imag(r.ν)) < 1e-10
                @test abs(real(ν) - real(r.ν)) / abs(real(r.ν)) < tol_rat
            end

            # Bref/Binc = 反射係数 (G(ω) = Bref/(2iω Binc) に使う比率)
            ratio_julia = amp.Bref / amp.Binc
            ratio_ref   = r.Bref   / r.Binc
            @test abs(ratio_julia - ratio_ref) / abs(ratio_ref) < tol_rat
        end
    end

end
