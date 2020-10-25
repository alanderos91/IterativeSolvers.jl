export lsmr, lsmr!

using LinearAlgebra

mutable struct LSMRIterable{T,Tr,matT,adjT,solT,vecT,numT}
    A::matT
    adjointA::adjT
    x::solT
    λ::numT

    # Bidiagonalization
    btmp::vecT
    α::T
    β::T
    u::vecT
    v::solT
    h::solT
    hbar::solT

    # Rotations
    ζbar::Tr
    αbar::Tr
    ρ::Tr
    ρbar::Tr
    cbar::Tr
    sbar::Tr

    # Estimation of ||r||.
    βdd::Tr
    βd::Tr
    ρdold::Tr
    τtildeold::Tr
    θtilde::Tr
    ζ::Tr
    d::Tr

    # Estimation of ||A|| and cond(A).
    normA::Tr
    condA::Tr
    normx::Tr
    normA2::Tr
    maxrbar::Tr
    minrbar::Tr

    # Worker arrays for 5-arg GEMM
    tmp_u::vecT
    tmp_v::solT

    # Bookkeeping
    mvps::Int
    mtvps::Int
    maxiter::Int
    normb::Tr
    normr::Tr
    normAr::Tr
    atol::Tr
    btol::Tr
    ctol::Tr
    istop::Int
end

@inline converged(it::LSMRIterable) = it.istop ∉ (3, 6, 7)

@inline start(it::LSMRIterable) = 0

@inline done(it::LSMRIterable, iteration::Int) = iteration ≥ it.maxiter || it.istop > 0

function reset_rhs!(it::LSMRIterable, b)
    # Initialize
    T = Adivtype(it.A, b)
    Tr = real(T)

    copy!(it.btmp, b)

    # form the first vectors u and v (satisfy β*u = b, α*v = A'u)
    mul!(it.tmp_u, it.A, it.x)
    @. it.u = b - it.tmp_u
    it.β = norm(it.u)
    @. it.u *= inv(it.β)
    
    mul!(it.v, it.adjointA, it.u)
    it.α = norm(it.v)
    @. it.v *= inv(it.α)

    # Initialize variables for 1st iteration.
    it.ζbar = it.α * it.β
    it.αbar = it.α
    it.ρ = one(Tr)
    it.ρbar = one(Tr)
    it.cbar = one(Tr)
    it.sbar = zero(Tr)

    copyto!(it.h, it.v)
    fill!(it.hbar, zero(Tr))

    # Initialize variables for estimation of ||r||.
    it.βdd = it.β
    it.βd = zero(Tr)
    it.ρdold = one(Tr)
    it.τtildeold = zero(Tr)
    it.θtilde = zero(Tr)
    it.ζ = zero(Tr)
    it.d = zero(Tr)

    # Initialize variables for estimation of ||A|| and cond(A).
    it.normA, it.condA, it.normx = -one(Tr), -one(Tr), -one(Tr)
    it.normA2 = abs2(it.α)
    it.maxrbar = zero(Tr)
    it.minrbar = 1e100

    # Items for use in stopping rules.
    it.normb = it.β
    it.normr = it.β
    it.normAr = it.α * it.β
    it.istop = it.normAr == 0 ? 1 : 0 # is this the correct exit code?

    # Other bookkeeping.
    it.mvps = 1
    it.mtvps = 1

    return it
end

function Base.iterate(it::LSMRIterable, iteration::Int=start(it))
    if done(it, iteration) return nothing end

    T = eltype(it.x)
    Tr = real(T)

    # Match a few names in the original code
    iter = iteration + 1
    istop = it.istop
    maxiter = it.maxiter
    atol = it.atol
    btol = it.btol
    ctol = it.ctol

    # Update u, v, α, β.
    mul!(it.tmp_u, it.A, it.v)
    @. it.u = it.tmp_u + it.u * -it.α
    it.β = norm(it.u)
    if it.β > 0
        @. it.u *= inv(it.β)
        mul!(it.tmp_v, it.adjointA, it.u)
        @. it.v = it.tmp_v + it.v * -it.β
        it.α = norm(it.v)
        @. it.v *= inv(it.α)
    end

    # Construct rotation Qhat_{k,2k+1}.
    αhat = hypot(it.αbar, it.λ)
    chat = it.αbar / αhat
    shat = it.λ / αhat

    # Use a plane rotation (Q_i) to turn B_i to R_i.
    ρold = it.ρ
    it.ρ = hypot(αhat, it.β)
    c = αhat / it.ρ
    s = it.β / it.ρ
    θnew = s * it.α
    it.αbar = c * it.α

    # Use a plane rotation (Qbar_i) to turn R_i^T to R_i^bar.
    ρbarold = it.ρbar
    ζold = it.ζ
    θbar = it.sbar * it.ρ
    ρtemp = it.cbar * it.ρ
    it.ρbar = hypot(it.cbar * it.ρ, θnew)
    it.cbar = it.cbar * it.ρ / it.ρbar
    it.sbar = θnew / it.ρbar
    it.ζ = it.cbar * it.ζbar
    it.ζbar = - it.sbar * it.ζbar

    # Update h, h_hat, x.
    @. it.hbar = it.hbar * (-θbar * it.ρ / (ρold * ρbarold)) + it.h
    @. it.x += (it.ζ / (it.ρ * it.ρbar)) * it.hbar
    @. it.h = it.h * (-θnew / it.ρ) + it.v

    ##############################################################################
    ##
    ## Estimate of ||r||
    ##
    ##############################################################################

    # Apply rotation Qhat_{k,2k+1}.
    βacute = chat * it.βdd
    βcheck = - shat * it.βdd

    # Apply rotation Q_{k,k+1}.
    βhat = c * βacute
    it.βdd = - s * βacute

    # Apply rotation Qtilde_{k-1}.
    θtildeold = it.θtilde
    ρtildeold = hypot(it.ρdold, θbar)
    ctildeold = it.ρdold / ρtildeold
    stildeold = θbar / ρtildeold
    it.θtilde = stildeold * it.ρbar
    it.ρdold = ctildeold * it.ρbar
    it.βd = - stildeold * it.βd + ctildeold * βhat

    it.τtildeold = (ζold - θtildeold * it.τtildeold) / ρtildeold
    τd = (it.ζ - it.θtilde * it.τtildeold) / it.ρdold
    it.d += abs2(βcheck)
    it.normr = sqrt(it.d + abs2(it.βd - τd) + abs2(it.βdd))

    # Estimate ||A||.
    it.normA2 += abs2(it.β)
    it.normA  = sqrt(it.normA2)
    it.normA2 += abs2(it.α)

    # Estimate cond(A).
    it.maxrbar = max(it.maxrbar, ρbarold)
    if iter > 1
        it.minrbar = min(it.minrbar, ρbarold)
    end
    it.condA = max(it.maxrbar, ρtemp) / min(it.minrbar, ρtemp)

    ##############################################################################
    ##
    ## Test for convergence
    ##
    ##############################################################################

    # Compute norms for convergence testing.
    it.normAr = abs(it.ζbar)
    it.normx = norm(it.x)

    # Now use these norms to estimate certain other quantities,
    # some of which will be small near a solution.
    test1 = it.normr / it.normb
    test2 = it.normAr / (it.normA * it.normr)
    test3 = inv(it.condA)
    # push!(log, :cnorm, test3)
    # push!(log, :anorm, test2)
    # push!(log, :rnorm, test1)
    # verbose && @printf("%3d\t%1.2e\t%1.2e\t%1.2e\n",iter,test2,test3,test1)

    t1 = test1 / (one(Tr) + it.normA * it.normx / it.normb)
    rtol = it.btol + it.atol * it.normA * it.normx / it.normb
    # The following tests guard against extremely small values of
    # atol, btol or ctol.  (The user may have set any or all of
    # the parameters atol, btol, conlim  to 0.)
    # The effect is equivalent to the normAl tests using
    # atol = eps,  btol = eps,  conlim = 1/eps.
    if iter >= maxiter istop = 7 end
    if 1 + test3 <= 1  istop = 6 end
    if 1 + test2 <= 1  istop = 5 end
    if 1 + t1 <= 1     istop = 4 end
    # Allow for tolerances set by the user.
    if test3 <= ctol   istop = 3 end
    if test2 <= atol   istop = 2 end
    if test1 <= rtol   istop = 1 end

    it.istop = istop

    cnorm = test3
    anorm = test2
    rnorm = test1

    return (rnorm, anorm, cnorm), iter
end

function lsmr_iterator!(x, A, b;
    atol::Number = 1e-6, btol::Number = 1e-6, conlim::Number = 1e8,
    maxiter::Int = maximum(size(A)), λ::Number = 0
    )
    #
    T = Adivtype(A, b)
    Tr = real(T)
    
    ctol = conlim > 0 ? convert(Tr, inv(conlim)) : zero(Tr)

    m, n = size(A)

    btmp = similar(b)
    u, v, h, hbar = similar(b, T), similar(x, T), similar(x, T), similar(x, T)
    tmp_u = similar(b)
    tmp_v = similar(v)

    # Construct iterator and initialize based on RHS.
    # reset_rhs! will correctly set scalar fields.
    it = LSMRIterable(
            A, adjoint(A), x, λ,
            # Bidiagonalization
            btmp, zero(T), zero(T), u, v, h, hbar,
            # Rotations
            zero(Tr), zero(Tr), zero(Tr), zero(Tr), zero(Tr), zero(Tr),
            # Estimation of ||r||
            zero(Tr), zero(Tr), zero(Tr), zero(Tr), zero(Tr), zero(Tr), zero(Tr),
            # Estimation of ||A|| and cond(A)
            zero(Tr), zero(Tr), zero(Tr), zero(Tr), zero(Tr), zero(Tr),
            # Worker arrays for 5-arg GEMM
            tmp_u, tmp_v,
            # Bookkeeping
            0, 0, maxiter, zero(Tr), zero(Tr), zero(Tr), Tr(atol), Tr(btol), Tr(ctol), 0
    )
    reset_rhs!(it, b)

    return it
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
    log::Bool=false, verbose::Bool = false, kwargs...
    )
    # Sanity-checking
    m = size(A, 1)
    n = size(A, 2)
    length(x) == n || error("x has length $(length(x)) but should have length $n")
    length(b) == m || error("b has length $(length(b)) but should have length $m")
    
    history = ConvergenceHistory(partial=!log)
    log && reserve!(history, [:anorm, :rnorm, :cnorm], maxiter)

    # Run LSMR
    iterable = lsmr_iterator!(x, A, b; maxiter=maxiter, kwargs...)
    
    history[:atol] = iterable.atol
    history[:btol] = iterable.btol
    history[:ctol] = iterable.ctol

    verbose && @printf("=== lsmr ===\n%4s\t%7s\t\t%7s\t\t%7s\n","iter","anorm","cnorm","rnorm")
    
    for (iteration, item) = enumerate(iterable)
        # item[1] = rnorm
        # item[2] = anorm
        # item[3] = cnorm
        if log
            nextiter!(history, mvps = 1, mtvps = 1)
            push!(history, :rnorm, item[1])
            push!(history, :anorm, item[2])
            push!(history, :cnorm, item[3])
        end
        if verbose
            @printf("%3d\t%1.2e\t%1.2e\t%1.2e\n",
                iteration, item[2], item[3], item[1])
        end
    end

    verbose && println()
    log && setconv(history, converged(iterable))
    log && shrink!(history)

    log ? (x, history) : x
end

#########################
# Method Implementation #
#########################

function lsmr_method!(log::ConvergenceHistory, x, A, b, v, h, hbar;
    atol::Number = 1e-6, btol::Number = 1e-6, conlim::Number = 1e8,
    maxiter::Int = maximum(size(A)), λ::Number = 0,
    verbose::Bool=false
    )
    verbose && @printf("=== lsmr ===\n%4s\t%7s\t\t%7s\t\t%7s\n","iter","anorm","cnorm","rnorm")

    # Sanity-checking
    m = size(A, 1)
    n = size(A, 2)
    length(x) == n || error("x has length $(length(x)) but should have length $n")
    length(v) == n || error("v has length $(length(v)) but should have length $n")
    length(h) == n || error("h has length $(length(h)) but should have length $n")
    length(hbar) == n || error("hbar has length $(length(hbar)) but should have length $n")
    length(b) == m || error("b has length $(length(b)) but should have length $m")

    T = Adivtype(A, b)
    Tr = real(T)
    normrs = Tr[]
    normArs = Tr[]
    conlim > 0 ? ctol = convert(Tr, inv(conlim)) : ctol = zero(Tr)
    # form the first vectors u and v (satisfy  β*u = b,  α*v = A'u)
    tmp_u = similar(b)
    tmp_v = similar(v)
    mul!(tmp_u, A, x)
    b .-= tmp_u
    u = b
    β = norm(u)
    u .*= inv(β)
    adjointA = adjoint(A)
    mul!(v, adjointA, u)
    α = norm(v)
    v .*= inv(α)

    log[:atol] = atol
    log[:btol] = btol
    log[:ctol] = ctol

    # Initialize variables for 1st iteration.
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
    minrbar = 1e100

    # Items for use in stopping rules.
    normb = β
    istop = 0
    normr = β
    normAr = α * β
    iter = 0
    # Exit if b = 0 or A'b = 0.

    log.mvps=1
    log.mtvps=1
    if normAr != 0
        while iter < maxiter
            nextiter!(log,mvps=1)
            iter += 1
            mul!(tmp_u, A, v)
            u .= tmp_u .+ u .* -α
            β = norm(u)
            if β > 0
                log.mtvps+=1
                u .*= inv(β)
                mul!(tmp_v, adjointA, u)
                v .= tmp_v .+ v .* -β
                α = norm(v)
                v .*= inv(α)
            end

            # Construct rotation Qhat_{k,2k+1}.
            αhat = hypot(αbar, λ)
            chat = αbar / αhat
            shat = λ / αhat

            # Use a plane rotation (Q_i) to turn B_i to R_i.
            ρold = ρ
            ρ = hypot(αhat, β)
            c = αhat / ρ
            s = β / ρ
            θnew = s * α
            αbar = c * α

            # Use a plane rotation (Qbar_i) to turn R_i^T to R_i^bar.
            ρbarold = ρbar
            ζold = ζ
            θbar = sbar * ρ
            ρtemp = cbar * ρ
            ρbar = hypot(cbar * ρ, θnew)
            cbar = cbar * ρ / ρbar
            sbar = θnew / ρbar
            ζ = cbar * ζbar
            ζbar = - sbar * ζbar

            # Update h, h_hat, x.
            hbar .= hbar .* (-θbar * ρ / (ρold * ρbarold)) .+ h
            x .+= (ζ / (ρ * ρbar)) * hbar
            h .= h .* (-θnew / ρ) .+ v

            ##############################################################################
            ##
            ## Estimate of ||r||
            ##
            ##############################################################################

            # Apply rotation Qhat_{k,2k+1}.
            βacute = chat * βdd
            βcheck = - shat * βdd

            # Apply rotation Q_{k,k+1}.
            βhat = c * βacute
            βdd = - s * βacute

            # Apply rotation Qtilde_{k-1}.
            θtildeold = θtilde
            ρtildeold = hypot(ρdold, θbar)
            ctildeold = ρdold / ρtildeold
            stildeold = θbar / ρtildeold
            θtilde = stildeold * ρbar
            ρdold = ctildeold * ρbar
            βd = - stildeold * βd + ctildeold * βhat

            τtildeold = (ζold - θtildeold * τtildeold) / ρtildeold
            τd = (ζ - θtilde * τtildeold) / ρdold
            d += abs2(βcheck)
            normr = sqrt(d + abs2(βd - τd) + abs2(βdd))

            # Estimate ||A||.
            normA2 += abs2(β)
            normA  = sqrt(normA2)
            normA2 += abs2(α)

            # Estimate cond(A).
            maxrbar = max(maxrbar, ρbarold)
            if iter > 1
                minrbar = min(minrbar, ρbarold)
            end
            condA = max(maxrbar, ρtemp) / min(minrbar, ρtemp)

            ##############################################################################
            ##
            ## Test for convergence
            ##
            ##############################################################################

            # Compute norms for convergence testing.
            normAr  = abs(ζbar)
            normx = norm(x)

            # Now use these norms to estimate certain other quantities,
            # some of which will be small near a solution.
            test1 = normr / normb
            test2 = normAr / (normA * normr)
            test3 = inv(condA)
            push!(log, :cnorm, test3)
            push!(log, :anorm, test2)
            push!(log, :rnorm, test1)
            verbose && @printf("%3d\t%1.2e\t%1.2e\t%1.2e\n",iter,test2,test3,test1)

            t1 = test1 / (one(Tr) + normA * normx / normb)
            rtol = btol + atol * normA * normx / normb
            # The following tests guard against extremely small values of
            # atol, btol or ctol.  (The user may have set any or all of
            # the parameters atol, btol, conlim  to 0.)
            # The effect is equivalent to the normAl tests using
            # atol = eps,  btol = eps,  conlim = 1/eps.
            if iter >= maxiter istop = 7; break end
            if 1 + test3 <= 1 istop = 6; break end
            if 1 + test2 <= 1 istop = 5; break end
            if 1 + t1 <= 1 istop = 4; break end
            # Allow for tolerances set by the user.
            if test3 <= ctol istop = 3; break end
            if test2 <= atol istop = 2; break end
            if test1 <= rtol  istop = 1; break end
        end
    end
    verbose && @printf("\n")
    setconv(log, istop ∉ (3, 6, 7))
    x
end
