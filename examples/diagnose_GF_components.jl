using Pkg; Pkg.activate("/Users/yusuke/work/Teukolsky.jl")
using Teukolsky, Plots, LaTeXStrings, Printf

# ── Parameters ───────────────────────────────────────────────
T = Float64
s, l, m = -2, 2, 2
a       = T(0.0)
r_src   = T(10.0)
nmax    = 200

N     = 500
ω_max = T(10.0)
Δω    = 2ω_max / N

# positive ω only (we'll mirror later if needed)
ω_pos = T[(n + 0.5) * Δω for n in 0:(N÷2 - 1)]

# ── Storage ──────────────────────────────────────────────────
Npos = length(ω_pos)
arr_Rup   = Vector{Complex{T}}(undef, Npos)
arr_Rdown = Vector{Complex{T}}(undef, Npos)
arr_Binc  = Vector{Complex{T}}(undef, Npos)
arr_Bref  = Vector{Complex{T}}(undef, Npos)
arr_nu    = Vector{Complex{T}}(undef, Npos)
arr_GF    = Vector{Complex{T}}(undef, Npos)
arr_Ap    = Vector{Complex{T}}(undef, Npos)
arr_Am    = Vector{Complex{T}}(undef, Npos)
arr_Rdown_raw = Vector{Complex{T}}(undef, Npos)

println("Computing components ($Npos points) ...")
ν_prev = nothing
for i in 1:Npos
    ω = ω_pos[i]
    amp = compute_amplitudes(s, l, m, a, ω; nmax=nmax)
    ν   = amp.ν
    arr_nu[i] = ν

    if !isfinite(real(ν)) || !isfinite(imag(ν))
        arr_Rup[i] = arr_Rdown[i] = arr_Binc[i] = arr_Bref[i] = arr_GF[i] = NaN + NaN*im
        arr_Ap[i] = arr_Am[i] = arr_Rdown_raw[i] = NaN + NaN*im
        continue
    end

    p = MSTParams(s, l, m, a, ω)

    arr_Binc[i] = amp.Binc
    arr_Bref[i] = amp.Bref
    arr_Ap[i]   = amp.Ap
    arr_Am[i]   = amp.Am

    arr_Rup[i]   = try Rup(p, ν, amp.fn, r_src; nmax=nmax) catch; NaN+NaN*im end
    arr_Rdown[i] = try Rdown(p, ν, amp.fn, r_src; nmax=nmax) catch; NaN+NaN*im end

    # raw R^ν_+ (before division by A+)
    arr_Rdown_raw[i] = try
        ϵ = p.ϵ; κ = p.κ
        phase = exp(-im * (ϵ * log(ϵ) - (1 - κ) / 2 * ϵ))
        norm_val = amp.Ap * ω^(-1) * phase
        arr_Rdown[i] * norm_val
    catch; NaN+NaN*im end

    # G = Bref * Rup / (2iω Binc) + Rdown / (2iω)
    arr_GF[i] = try
        amp.Bref * arr_Rup[i] / (2im * ω * amp.Binc) + arr_Rdown[i] / (2im * ω)
    catch
        NaN + NaN*im
    end

    ν_prev = ν
    i % 100 == 0 && (print("."); flush(stdout))
end
println(" done")

# ── Plots ────────────────────────────────────────────────────
ω_arr = Float64.(ω_pos)

p1 = plot(ω_arr, abs.(arr_Rup);   yscale=:log10, label=L"|R_{\rm up}|",   xlabel=L"\omega", title="Rup", framestyle=:box)
p2 = plot(ω_arr, abs.(arr_Rdown); yscale=:log10, label=L"|R_{\rm down}|", xlabel=L"\omega", title="Rdown", framestyle=:box)
p3 = plot(ω_arr, abs.(arr_Binc);  yscale=:log10, label=L"|B^{\rm inc}|",  xlabel=L"\omega", title="Binc", framestyle=:box)
p4 = plot(ω_arr, abs.(arr_Bref);  yscale=:log10, label=L"|B^{\rm ref}|",  xlabel=L"\omega", title="Bref", framestyle=:box)
p5 = plot(ω_arr, real.(arr_nu);   label=L"\mathrm{Re}(\nu)", xlabel=L"\omega", title="ν", framestyle=:box)
plot!(p5, ω_arr, imag.(arr_nu);   label=L"\mathrm{Im}(\nu)")
p6 = plot(ω_arr, abs.(arr_GF);    yscale=:log10, label=L"|G(\omega)|", xlabel=L"\omega", title="G(ω)", framestyle=:box)

fig = plot(p1, p2, p3, p4, p5, p6; layout=(3,2), size=(1000, 900), dpi=150)
savefig(fig, joinpath(@__DIR__, "diagnose_GF_components.png"))
println("Saved: diagnose_GF_components.png")

# Additional diagnostic: Rdown decomposition
p7 = plot(ω_arr, abs.(arr_Rdown_raw); yscale=:log10, label=L"|R^{\nu}_+|\ \mathrm{(raw)}", xlabel=L"\omega", title="Rdown raw vs normalized", framestyle=:box)
plot!(p7, ω_arr, abs.(arr_Rdown); yscale=:log10, label=L"|R_{\rm down}|\ \mathrm{(normalized)}")
p8 = plot(ω_arr, abs.(arr_Ap); yscale=:log10, label=L"|A^{\nu}_+|", xlabel=L"\omega", title="A+ and A−", framestyle=:box)
plot!(p8, ω_arr, abs.(arr_Am); yscale=:log10, label=L"|A^{\nu}_-|")

fig2 = plot(p7, p8; layout=(2,1), size=(800, 600), dpi=150)
savefig(fig2, joinpath(@__DIR__, "diagnose_Rdown_detail.png"))
println("Saved: diagnose_Rdown_detail.png")

# Zoom into QNM region: check G decomposition
mask = ω_arr .< 2.0
ω_z = ω_arr[mask]

# G = Bref * Rup / (2iω Binc) + Rdown / (2iω)
term1 = arr_Bref[mask] .* arr_Rup[mask] ./ (2im .* ω_z .* arr_Binc[mask])
term2 = arr_Rdown[mask] ./ (2im .* ω_z)

pz1 = plot(ω_z, abs.(arr_Binc[mask]); yscale=:log10, label=L"|B^{\rm inc}|",
    xlabel=L"\omega", title="Binc near QNM", framestyle=:box, xlim=(0, 2))
pz2 = plot(ω_z, abs.(term1); yscale=:log10, label=L"|B^{\rm ref}R_{\rm up}/(2i\omega B^{\rm inc})|",
    xlabel=L"\omega", title="G(ω) decomposition", framestyle=:box, xlim=(0, 2))
plot!(pz2, ω_z, abs.(term2); yscale=:log10, label=L"|R_{\rm down}/(2i\omega)|")
plot!(pz2, ω_z, abs.(arr_GF[mask]); yscale=:log10, label=L"|G(\omega)|", ls=:dash)
pz3 = plot(ω_z, abs.(arr_Rup[mask]); yscale=:log10, label=L"|R_{\rm up}|",
    xlabel=L"\omega", title="Rup, Rdown near QNM", framestyle=:box, xlim=(0, 2))
plot!(pz3, ω_z, abs.(arr_Rdown[mask]); yscale=:log10, label=L"|R_{\rm down}|")
pz4 = plot(ω_z, abs.(arr_Bref[mask]); yscale=:log10, label=L"|B^{\rm ref}|",
    xlabel=L"\omega", title="Bref near QNM", framestyle=:box, xlim=(0, 2))

fig3 = plot(pz1, pz2, pz3, pz4; layout=(2,2), size=(1000, 700), dpi=150)
savefig(fig3, joinpath(@__DIR__, "diagnose_QNM_zoom.png"))
println("Saved: diagnose_QNM_zoom.png")
display(fig2)
