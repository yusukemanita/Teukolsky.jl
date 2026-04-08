# ============================================================
#  Hypergeometric functions for MST radial solutions
#
#  - Gauss 2F1 for Rin (via HypergeometricFunctions.jl + recurrence)
#  - Tricomi U for Rup (via Kummer relation + recurrence)
#
#  Teukolsky parameters:
#    2F1: aF = ν+1-iτ, bF = -ν-iτ, cF = 1-s-i(ε+τ)
#    U:   aU = ν+s+1-iε, bU = 2ν+2, argument c = -2i·ẑ
# ============================================================

using HypergeometricFunctions: _₂F₁, _₁F₁

# ============================================================
#  HypergeometricU via Kummer relation
#  U(a,b,z) = Γ(1-b)/Γ(a+1-b) M(a,b,z)
#           + Γ(b-1)/Γ(a) z^(1-b) M(a+1-b, 2-b, z)
# ============================================================

function hypergeometric_U(a, b, z)
    # For near-integer b, add small perturbation to avoid Gamma poles
    b_int = round(Int, real(b))
    if abs(b - b_int) < 1e-8
        b = b + 1e-8im
    end

    term1 = gamma(complex(1 - b)) / gamma(complex(a + 1 - b)) * _₁F₁(a, b, z)
    term2 = gamma(complex(b - 1)) / gamma(complex(a)) * z^(1 - b) * _₁F₁(a + 1 - b, 2 - b, z)
    return term1 + term2
end

# ============================================================
#  2F1 evaluation with DLMF recurrence
#  Follows MST.m lines 139-167
# ============================================================

struct H2F1Params
    aF::ComplexF64  # ν + 1 - iτ
    bF::ComplexF64  # -ν - iτ
    cF::ComplexF64  # 1 - s - i(ε+τ)
    x::ComplexF64   # (r+ - r) / (2κ)
end

function H2F1Params(p::MSTParams, ν, x)
    aF = ν + 1 - im * p.τ
    bF = -ν - im * p.τ
    cF = 1 - p.s - im * (p.ϵ + p.τ)
    H2F1Params(aF, bF, cF, complex(x))
end

function h2f1_exact(hp::H2F1Params, n::Int)
    _₂F₁(n + hp.aF, hp.bF - n, hp.cF, hp.x)
end

"""
3-term recurrence for 2F1 going upward in n (n >= 2).
Returns (term1, term2) such that H2F1[n] = term1 + term2.
Requires H2F1[n-2] and H2F1[n-1] already computed.
"""
function h2f1_up(hp::H2F1Params, n::Int, h2f1_nm2, h2f1_nm1)
    a, b, c, x = hp.aF, hp.bF, hp.cF, hp.x
    denom = (3 - a + b - 2n) * (1 + b - c - n) * (-1 + a + n)
    t1 = -(1 - a + b - 2n) * (1 + b - n) * (-1 + a - c + n) * h2f1_nm2
    t2 = -((-2 + a - b + 2n) * (-2 + 2a - 2b + 2a*b + c - a*c - b*c +
          4n - 2a*n + 2b*n - 2n^2 + 3x - 4a*x + a^2*x + 4b*x -
          2a*b*x + b^2*x - 8n*x + 4a*n*x - 4b*n*x + 4n^2*x)) * h2f1_nm1
    return t1 / denom, t2 / denom
end

"""
3-term recurrence for 2F1 going downward in n (n <= -1).
Returns (term1, term2) such that H2F1[n] = term1 + term2.
Requires H2F1[n+2] and H2F1[n+1] already computed.
"""
function h2f1_down(hp::H2F1Params, n::Int, h2f1_np2, h2f1_np1)
    a, b, c, x = hp.aF, hp.bF, hp.cF, hp.x
    denom = (-3 - a + b - 2n) * (-1 + b - n) * (1 + a - c + n)
    t1 = (-2 - a + b - 2n) * (-2 - 2a + 2b + 2a*b + c - a*c - b*c -
          4n - 2a*n + 2b*n - 2n^2 + 3x + 4a*x + a^2*x - 4b*x -
          2a*b*x + b^2*x + 8n*x + 4a*n*x - 4b*n*x + 4n^2*x) * h2f1_np1
    t2 = -(-1 - a + b - 2n) * (-1 + b - c - n) * (1 + a + n) * h2f1_np2
    return t1 / denom, t2 / denom
end

# ============================================================
#  2F1 derivative: d/dx [₂F₁(n+a, b-n, c, x)] = (n+a)(b-n)/c · ₂F₁(n+a+1, b-n+1, c+1, x)
# ============================================================

function dh2f1_exact(hp::H2F1Params, n::Int)
    (n + hp.aF) * (hp.bF - n) / hp.cF *
        _₂F₁(n + hp.aF + 1, hp.bF - n + 1, hp.cF + 1, hp.x)
end

"""
3-term recurrence for d(2F1)/dx going upward in n.
Returns (term1, term2, term3) where term3 is the H2F1 mixing term.
"""
function dh2f1_up(hp::H2F1Params, n::Int, dh2f1_nm2, dh2f1_nm1, h2f1_nm1)
    a, b, c, x = hp.aF, hp.bF, hp.cF, hp.x
    denom_outer = (1 + b - c - n) * (-1 + a + n)
    coeff_nm2 = -((1 - a + b - 2n) * (1 + b - n) * (-1 + a - c + n)) / (3 - a + b - 2n)
    poly = (-2 + 2a - 2b + 2a*b + c - a*c - b*c + 4n - 2a*n + 2b*n - 2n^2 +
            3x - 4a*x + a^2*x + 4b*x - 2a*b*x + b^2*x - 8n*x + 4a*n*x - 4b*n*x + 4n^2*x)
    coeff_nm1 = (2 - a + b - 2n) * poly / (3 - a + b - 2n)
    coeff_h = (1 - a + b - 2n) * (2 - a + b - 2n)
    return coeff_nm2 * dh2f1_nm2 / denom_outer,
           coeff_nm1 * dh2f1_nm1 / denom_outer,
           coeff_h * h2f1_nm1 / denom_outer
end

"""
3-term recurrence for d(2F1)/dx going downward in n.
Returns (term1, term2, term3) where term3 is the H2F1 mixing term.
"""
function dh2f1_down(hp::H2F1Params, n::Int, dh2f1_np2, dh2f1_np1, h2f1_np1)
    a, b, c, x = hp.aF, hp.bF, hp.cF, hp.x
    denom_outer = (-1 + b - n) * (1 + a - c + n)
    poly = (-2 - 2a + 2b + 2a*b + c - a*c - b*c - 4n - 2a*n + 2b*n - 2n^2 +
            3x + 4a*x + a^2*x - 4b*x - 2a*b*x + b^2*x + 8n*x + 4a*n*x - 4b*n*x + 4n^2*x)
    coeff_np1 = (-2 - a + b - 2n) * poly / (-3 - a + b - 2n)
    coeff_np2 = -((-1 - a + b - 2n) * (-1 + b - c - n) * (1 + a + n)) / (-3 - a + b - 2n)
    coeff_h = (-2 - a + b - 2n) * (-1 - a + b - 2n)
    return coeff_np1 * dh2f1_np1 / denom_outer,
           coeff_np2 * dh2f1_np2 / denom_outer,
           coeff_h * h2f1_np1 / denom_outer
end

# ============================================================
#  HypergeometricU evaluation with recurrence
#  HU[n] = c^n * U(n+aU, 2n+bU, c)
#  where aU = ν+s+1-iε, bU = 2ν+2, c = -2i·ẑ
# ============================================================

struct HUParams
    aU::ComplexF64  # ν + s + 1 - iε
    bU::ComplexF64  # 2ν + 2
    c::ComplexF64   # -2i * ẑ
end

function HUParams(p::MSTParams, ν, zhat)
    aU = ν + p.s + 1 - im * p.ϵ
    bU = 2ν + 2
    c = -2im * zhat
    HUParams(aU, bU, c)
end

function hu_exact(hp::HUParams, n::Int)
    hp.c^n * hypergeometric_U(n + hp.aU, 2n + hp.bU, hp.c)
end

"""
3-term recurrence for HU going upward in n (n >= 2).
Returns (term1, term2) such that HU[n] = term1 + term2.
"""
function hu_up(hp::HUParams, n::Int, hu_nm2, hu_nm1)
    a, b, c = hp.aU, hp.bU, hp.c
    denom = (-1 + a + n) * (-4 + b + 2n)
    t1 = (-2 - a + b + n) * (-2 + b + 2n) * hu_nm2
    t2 = (-3 + b + 2n) * (8 + (b + 2n)^2 + 2(a + n)*c - (b + 2n)*(6 + c)) * hu_nm1 / c
    return t1 / denom, t2 / denom
end

"""
3-term recurrence for HU going downward in n (n <= -1).
Returns (term1, term2) such that HU[n] = term1 + term2.
"""
function hu_down(hp::HUParams, n::Int, hu_np2, hu_np1)
    a, b, c = hp.aU, hp.bU, hp.c
    denom = (-a + b + n) * (2 + b + 2n)
    t1 = -((1 + b + 2n) * (b^2 + 4n*(1 + n) + b*(2 + 4n - c) + 2a*c)) * hu_np1 / c
    t2 = (1 + a + n) * (b + 2n) * hu_np2
    return t1 / denom, t2 / denom
end

# ============================================================
#  HypergeometricU derivative with respect to ẑ
#  dHU[n] = d/dẑ [c^n U(n+a, 2n+b, c)]  with dc/dẑ = -2i
# ============================================================

function dhu_exact(hp::HUParams, n::Int)
    a, b, c = hp.aU, hp.bU, hp.c
    -2im * (c^(n-1) * n * hypergeometric_U(a + n, b + 2n, c) -
            c^n * (a + n) * hypergeometric_U(1 + a + n, 1 + b + 2n, c))
end

"""
3-term recurrence for dHU going upward in n (n >= 2).
Returns (term1, term2, term3) where term3 is the HU mixing term.
"""
function dhu_up(hp::HUParams, n::Int, dhu_nm2, dhu_nm1, hu_nm1)
    a, b, c = hp.aU, hp.bU, hp.c
    denom = -1 + a + n
    t1 = (-2 - a + b + n) * (-2 + b + 2n) * dhu_nm2 / (-4 + b + 2n)
    poly = 8 + b^2 + 4(-3 + n)*n + b*(-6 + 4n - c) + 2a*c
    t2 = (-3 + b + 2n) * poly * dhu_nm1 / ((-4 + b + 2n) * c)
    t3 = 2im * (-3 + b + 2n) * (-2 + b + 2n) * hu_nm1 / c^2
    return t1 / denom, t2 / denom, t3 / denom
end

"""
3-term recurrence for dHU going downward in n (n <= -1).
Returns (term1, term2, term3) where term3 is the HU mixing term.
"""
function dhu_down(hp::HUParams, n::Int, dhu_np2, dhu_np1, hu_np1)
    a, b, c = hp.aU, hp.bU, hp.c
    denom = (a - b - n) * (2 + b + 2n) * c^2
    poly = b^2 + 4n*(1 + n) + b*(2 + 4n - c) + 2a*c
    t1 = (1 + b + 2n) * c * poly * dhu_np1
    t2 = (b + 2n) * (-(1 + a + n) * c^2) * dhu_np2
    t3 = (b + 2n) * 2im * (1 + b + 2n) * (2 + b + 2n) * hu_np1
    return t1 / denom, t2 / denom, t3 / denom
end
