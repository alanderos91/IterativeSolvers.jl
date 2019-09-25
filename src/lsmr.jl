export lsmr, lsmr!

using LinearAlgebra

mutable struct LSMRIterable{matT, adjointT, bvecT <: AbstractVector, xvecT <: AbstractVector, vecAdivT <: AbstractVector, numT <: Real, λnumT <: Real}
    A::matT
    b::bvecT
    x::xvecT
    λ::λnumT

    adjointA::adjointT
    u::bvecT
    v::vecAdivT
    α::numT
    β::numT

    tmp_u::bvecT
    tmp_v::vecAdivT

    h::vecAdivT
    hbar::vecAdivT

    ζbar::numT
    αbar::numT
    ρ::numT
    ρbar::numT
    cbar::numT
    sbar::numT

    # variables for estimation of ||r||.
    βdd::numT
    βd::numT
    ρdold::numT
    τtildeold::numT
    θtilde::numT
    ζ::numT
    d::numT

    # variables for estimation of ||A|| and cond(A).
    normA::numT
    condA::numT
    normx::numT
    normA2::numT
    maxrbar::numT
    minrbar::numT

    # variables for stopping rules
    normb::numT
    istop::Int
    normr::numT
    normAr::numT

    # user settings
    maxiter::Int
    atol::numT
    btol::numT
    ctol::numT
end

function lsmr_iterable!(x, A, b;
    atol::Number = 1e-6, btol::Number = 1e-6, conlim::Number = 1e8,
    maxiter::Int = maximum(size(A)), λ::Number = 0)
    # extract type information
    T = Adivtype(A, b)
    Tr = real(T)

    # vector allocations
    u = similar(b, T)
    copyto!(u, b)

    v, h, hbar = similar(x, T), similar(x, T), similar(x, T)

    # Sanity-checking
    m = size(A, 1)
    n = size(A, 2)
    length(x) == n || error("x has length $(length(x)) but should have length $n")
    length(v) == n || error("v has length $(length(v)) but should have length $n")
    length(h) == n || error("h has length $(length(h)) but should have length $n")
    length(hbar) == n || error("hbar has length $(length(hbar)) but should have length $n")
    length(b) == m || error("b has length $(length(b)) but should have length $m")

    conlim > 0 ? ctol = convert(Tr, inv(conlim)) : ctol = zero(Tr)

    # form the first vectors u and v (satisfy  β*u = b,  α*v = A'u)
    tmp_u = similar(b)
    tmp_v = similar(v)
    adjointA = adjoint(A)

    u, tmp_u, v, tmp_v, α, β = lsmr_initialize_u_and_v!(u, tmp_u, v, tmp_v, A, adjointA, x)

    # initialize variables for first iteration
    ζbar = α * β
    αbar = α
    ρ = one(Tr)
    ρbar = one(Tr)
    cbar = one(Tr)
    sbar = zero(Tr)

    copyto!(h, v)
    fill!(hbar, zero(Tr))

    # Initialize variables for estimation of ||r||.
    βdd = β
    βd = zero(Tr)
    ρdold = one(Tr)
    τtildeold = zero(Tr)
    θtilde  = zero(Tr)
    ζ = zero(Tr)
    d = zero(Tr)

    # Initialize variables for estimation of ||A|| and cond(A).
    normA, condA, normx = -one(Tr), -one(Tr), -one(Tr)
    normA2 = abs2(α)
    maxrbar = zero(Tr)
    minrbar = Tr(1e100)

    # Items for use in stopping rules.
    normb = β
    istop = 0
    normr = β
    normAr = α * β

    LSMRIterable(
        A, b, x, λ,   # variables in problem statement
        adjointA,
        u, v, α, β,   # variables for ???
        tmp_u, tmp_v, # intermediates for mat-vec product
        h, hbar,      # ???
        ζbar, αbar, ρ, ρbar, cbar, sbar, # ???
        # variables for estimation of ||r||.
        βdd, βd, ρdold, τtildeold, θtilde, ζ, d,
        # variables for estimation of ||A|| and cond(A).
        normA, condA, normx, normA2, maxrbar, minrbar,
        # variables for stopping rules
        normb, istop, normr, normAr,
        # user settings
        maxiter, Tr(atol), Tr(btol), ctol
    )
end

"""
    lsmr(A, b; kwrags...) -> x, [history]

Same as [`lsmr!`](@ref), but allocates a solution vector `x` initialized with zeros.
"""
lsmr(A, b; kwargs...) = lsmr!(zerox(A, b), A, b; kwargs...)

"""
    lsmr!(x, A, b; kwargs...) -> x, [history]

Minimizes ``\\|Ax - b\\|^2 + \\|λx\\|^2`` in the Euclidean norm. If multiple solutions
exists the minimum norm solution is returned.

The method is based on the Golub-Kahan bidiagonalization process. It is
algebraically equivalent to applying MINRES to the normal equations
``(A^*A + λ^2I)x = A^*b``, but has better numerical properties,
especially if ``A`` is ill-conditioned.

# Arguments
- `x`: Initial guess, will be updated in-place;
- `A`: linear operator;
- `b`: right-hand side.

## Keywords

- `λ::Number = 0`: lambda.
- `atol::Number = 1e-6`, `btol::Number = 1e-6`: stopping tolerances. If both are
  1.0e-9 (say), the final residual norm should be accurate to about 9 digits.
  (The final `x` will usually have fewer correct digits,
  depending on `cond(A)` and the size of damp).
- `conlim::Number = 1e8`: stopping tolerance. `lsmr` terminates if an estimate
  of `cond(A)` exceeds conlim.  For compatible systems Ax = b,
  conlim could be as large as 1.0e+12 (say).  For least-squares
  problems, conlim should be less than 1.0e+8.
  Maximum precision can be obtained by setting
- `atol` = `btol` = `conlim` = zero, but the number of iterations
  may then be excessive.
- `maxiter::Int = maximum(size(A))`: maximum number of iterations.
- `log::Bool`: keep track of the residual norm in each iteration;
- `verbose::Bool`: print convergence information during the iterations.

# Return values

**if `log` is `false`**

- `x`: approximated solution.

**if `log` is `true`**

- `x`: approximated solution.
- `ch`: convergence history.

**ConvergenceHistory keys**

- `:atol` => `::Real`: atol stopping tolerance.
- `:btol` => `::Real`: btol stopping tolerance.
- `:ctol` => `::Real`: ctol stopping tolerance.
- `:anorm` => `::Real`: anorm.
- `:rnorm` => `::Real`: rnorm.
- `:cnorm` => `::Real`: cnorm.
- `:resnom` => `::Vector`: residual norm at each iteration.
"""
function lsmr!(x, A, b;
    maxiter::Int = maximum(size(A)),
    log::Bool=false, verbose::Bool=false, kwargs...
    )
    history = ConvergenceHistory(partial=!log)
    reserve!(history, [:anorm,:rnorm,:cnorm], maxiter)

    iterable = lsmr_iterable!(x, A, b; maxiter = maxiter, kwargs...)

    history[:atol] = iterable.atol
    history[:btol] = iterable.btol
    history[:ctol] = iterable.ctol

    lsmr_method!(history, iterable, verbose)

    log && shrink!(history)
    log ? (x, history) : x
end

#########################
# Method Implementation #
#########################

function lsmr_method!(log::ConvergenceHistory, p::LSMRIterable,
    verbose
    )
    verbose && @printf("=== lsmr ===\n%4s\t%7s\t\t%7s\t\t%7s\n","iter","anorm","cnorm","rnorm")

    T = Adivtype(p.A, p.b)
    Tr = real(T)

    iter = 0
    # Exit if b = 0 or A'b = 0.

    log.mvps=1
    log.mtvps=1

    if p.normAr != 0
        while iter < p.maxiter
            nextiter!(log, mvps=1)
            iter += 1

            # update u and v
            lsmr_update_u_and_v!(log, p)

            # Construct rotation Qhat_{k,2k+1}.
            αhat = hypot(p.αbar, p.λ)
            chat = p.αbar / αhat
            shat = p.λ / αhat

            # Use a plane rotation (Q_i) to turn B_i to R_i.
            ρold = p.ρ
            p.ρ = hypot(αhat, p.β)
            c = αhat / p.ρ
            s = p.β / p.ρ
            θnew = s * p.α
            p.αbar = c * p.α

            # Use a plane rotation (Qbar_i) to turn R_i^T to R_i^bar.
            ρbarold = p.ρbar
            ζold = p.ζ
            θbar = p.sbar * p.ρ
            ρtemp = p.cbar * p.ρ
            p.ρbar = hypot(p.cbar * p.ρ, θnew)
            p.cbar = p.cbar * p.ρ / p.ρbar
            p.sbar = θnew / p.ρbar
            p.ζ = p.cbar * p.ζbar
            p.ζbar = - p.sbar * p.ζbar

            # Update h, h_hat, x.
            p.hbar .= p.hbar .* (-θbar * p.ρ / (ρold * ρbarold)) .+ p.h
            p.x .+= (p.ζ / (p.ρ * p.ρbar)) * p.hbar
            p.h .= p.h .* (-θnew / p.ρ) .+ p.v

            ##############################################################################
            ##
            ## Estimate of ||r||
            ##
            ##############################################################################

            # Apply rotation Qhat_{k,2k+1}.
            βacute = chat * p.βdd
            βcheck = - shat * p.βdd

            # Apply rotation Q_{k,k+1}.
            βhat = c * βacute
            p.βdd = - s * βacute

            # Apply rotation Qtilde_{k-1}.
            θtildeold = p.θtilde
            ρtildeold = hypot(p.ρdold, θbar)
            ctildeold = p.ρdold / ρtildeold
            stildeold = θbar / ρtildeold
            p.θtilde = stildeold * p.ρbar
            p.ρdold = ctildeold * p.ρbar
            p.βd = - stildeold * p.βd + ctildeold * βhat

            p.τtildeold = (ζold - θtildeold * p.τtildeold) / ρtildeold
            τd = (p.ζ - p.θtilde * p.τtildeold) / p.ρdold
            p.d += abs2(βcheck)
            p.normr = sqrt(p.d + abs2(p.βd - τd) + abs2(p.βdd))

            # Estimate ||A||.
            p.normA2 += abs2(p.β)
            p.normA  = sqrt(p.normA2)
            p.normA2 += abs2(p.α)

            # Estimate cond(A).
            p.maxrbar = max(p.maxrbar, ρbarold)
            if iter > 1
                p.minrbar = min(p.minrbar, ρbarold)
            end
            p.condA = max(p.maxrbar, ρtemp) / min(p.minrbar, ρtemp)

            ##############################################################################
            ##
            ## Test for convergence
            ##
            ##############################################################################

            # Compute norms for convergence testing.
            p.normAr  = abs(p.ζbar)
            p.normx = norm(p.x)

            # Now use these norms to estimate certain other quantities,
            # some of which will be small near a solution.
            test1 = p.normr / p.normb
            test2 = p.normAr / (p.normA * p.normr)
            test3 = inv(p.condA)
            push!(log, :cnorm, test3)
            push!(log, :anorm, test2)
            push!(log, :rnorm, test1)
            verbose && @printf("%3d\t%1.2e\t%1.2e\t%1.2e\n",iter,test2,test3,test1)

            t1 = test1 / (one(Tr) + p.normA * p.normx / p.normb)
            rtol = p.btol + p.atol * p.normA * p.normx / p.normb
            # The following tests guard against extremely small values of
            # atol, btol or ctol.  (The user may have set any or all of
            # the parameters atol, btol, conlim  to 0.)
            # The effect is equivalent to the normAl tests using
            # atol = eps,  btol = eps,  conlim = 1/eps.
            if iter >= p.maxiter p.istop = 7; break end
            if 1 + test3 <= 1 p.istop = 6; break end
            if 1 + test2 <= 1 p.istop = 5; break end
            if 1 + t1 <= 1 p.istop = 4; break end
            # Allow for tolerances set by the user.
            if test3 <= p.ctol p.istop = 3; break end
            if test2 <= p.atol p.istop = 2; break end
            if test1 <= rtol  p.istop = 1; break end
        end
    end
    verbose && @printf("\n")
    setconv(log, p.istop ∉ (3, 6, 7))
    p.x
end

function lsmr_initialize_u_and_v!(u, tmp_u, v, tmp_v, A, adjointA, x)
    mul!(tmp_u, A, x)
    u .-= tmp_u
    β = norm(u)
    u .*= inv(β)

    mul!(v, adjointA, u)
    α = norm(v)
    v .*= inv(α)

    return u, tmp_u, v, tmp_v, α, β
end

function lsmr_update_u_and_v!(log, p::LSMRIterable)
    mul!(p.tmp_u, p.A, p.v)
    p.u .= p.tmp_u .+ p.u .* -(p.α)
    p.β = norm(p.u)

    if p.β > 0
        log.mtvps += 1
        p.u .*= inv(p.β)
        mul!(p.tmp_v, p.adjointA, p.u)
        p.v .= p.tmp_v .+ p.v .* -(p.β)
        p.α = norm(p.v)
        p.v .*= inv(p.α)
    end

    return nothing
end
