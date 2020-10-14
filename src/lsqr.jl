export lsqr, lsqr!

mutable struct LSQRIterable{T,Tr,matT,adjT,solT,vecT,numT}
    A::matT
    adjointA::adjT
    x::solT
    damp::numT
    dampsq::numT

    # Bidiagonalization
    alpha::T
    beta::T
    u::vecT
    v::solT

    # Orthogonalization
    rhobar::Tr
    phibar::Tr
    cs2::Tr
    sn2::Tr

    # Forward substitution
    w::solT
    wrho::solT

    # Stopping criterion
    Anorm::Tr
    Acond::Tr
    ddnorm::Tr
    xxnorm::Tr
    res2::Tr
    z::Tr

    # Worker arrays for 5-arg GEMM
    tmpm::vecT
    tmpn::solT

    # Bookkeeping
    mvps::Int
    mtvps::Int
    maxiter::Int
    bnorm::Tr
    atol::Tr
    btol::Tr
    ctol::Tr
    istop::Int
end

@inline converged(it::LSQRIterable) = it.istop > 0

@inline start(it::LSQRIterable) = 0

@inline done(it::LSQRIterable, iteration::Int) = iteration ≥ it.maxiter || converged(it)

function reset_rhs!(it::LSQRIterable, b)
    # Initialize
    T = Adivtype(it.A, b)
    Tr = real(T)

    it.istop  = 0
    it.Anorm  = zero(Tr)
    it.Acond  = zero(Tr)
    it.ddnorm = zero(Tr)
    it.res2   = zero(Tr)
    it.xxnorm = zero(Tr)
    it.z      = zero(Tr)
    it.sn2    = zero(Tr)
    it.cs2    = -one(Tr)

    # Set up the first vectors u and v for the bidiagonalization.
    # These satisfy  beta*u = b-A*x,  alpha*v = A'u.
    mul!(it.u, it.A, it.x); @. it.u = b - it.u
    copyto!(it.v, it.x)
    it.beta = norm(it.u)
    if it.beta > 0
        @. it.u = it.u * inv(it.beta)
        mul!(it.v, it.adjointA, it.u)
        it.alpha = norm(it.v)
    end
    if it.alpha > 0
        @. it.v *= inv(it.alpha)
    end
    copyto!(it.w, it.v)

    Arnorm = it.alpha * it.beta
    if Arnorm == 0
        it.istop = 2
    end

    it.rhobar = it.alpha
    it.phibar = it.beta
    it.bnorm  = it.beta
    it.mvps   = 1
    it.mtvps  = 1

    return it
end

function iterate(it::LSQRIterable, iteration::Int=start(it))
    if done(it, iteration) return nothing end

    # Perform the next step of the bidiagonalization to obtain the
    # next beta, u, alpha, v.  These satisfy the relations
    #      beta*u  =  A*v  - alpha*u,
    #      alpha*v  =  A'*u - beta*v.

    # Note that the following three lines are a band aid for a GEMM: X: C := αAB + βC.
    # This is already supported in mul! for sparse and distributed matrices, but not yet dense
    mul!(it.tmpm, it.A, it.v)
    @. it.u = -it.alpha * it.u + it.tmpm
    it.beta = norm(it.u)
    if it.beta > 0
        # log.mtvps+=1
        @. it.u *= inv(it.beta)
        it.Anorm = sqrt(abs2(it.Anorm) + abs2(it.alpha) + abs2(it.beta) + it.dampsq)
        # Note that the following three lines are a band aid for a GEMM: X: C := αA'B + βC.
        # This is already supported in mul! for sparse and distributed matrices, but not yet dense
        mul!(it.tmpn, it.adjointA, it.u)
        @. it.v = -it.beta * it.v + it.tmpn
        it.alpha = norm(it.v)
        if it.alpha > 0
            @. it.v *= inv(it.alpha)
        end
    end

    # Use a plane rotation to eliminate the damping parameter.
    # This alters the diagonal (rhobar) of the lower-bidiagonal matrix.
    rhobar1   = sqrt(abs2(it.rhobar) + it.dampsq)
    cs1       = it.rhobar / rhobar1
    sn1       = it.damp   / rhobar1
    psi       = sn1 * it.phibar
    it.phibar = cs1 * it.phibar

    # Use a plane rotation to eliminate the subdiagonal element (beta)
    # of the lower-bidiagonal matrix, giving an upper-bidiagonal matrix.
    rho       =   sqrt(abs2(rhobar1) + abs2(it.beta))
    cs        =   rhobar1/rho
    sn        =   it.beta/rho
    theta     =   sn * it.alpha
    it.rhobar = - cs * it.alpha
    phi       =   cs * it.phibar
    it.phibar =   sn * it.phibar
    tau       =   sn * phi
    
    # Update x and w
    t1 =   phi  /rho
    t2 = - theta/rho

    @. it.x    += t1 * it.w
    @. it.w     = t2 * it.w + it.v
    @. it.wrho  = it.w * inv(rho)
    it.ddnorm  += norm(it.wrho)

    # Use a plane rotation on the right to eliminate the
    # super-diagonal element (theta) of the upper-bidiagonal matrix.
    # Then use the result to estimate  norm(x).
    delta      =   it.sn2 * rho
    gambar     =  -it.cs2 * rho
    rhs        =   phi - delta * it.z
    zbar       =   rhs / gambar
    xnorm      =   sqrt(it.xxnorm + abs2(zbar))
    gamma      =   sqrt(abs2(gambar) + abs2(theta))
    it.cs2     =   gambar / gamma
    it.sn2     =   theta  / gamma
    it.z       =   rhs    / gamma
    it.xxnorm +=   abs2(it.z)

    # Test for convergence.
    # First, estimate the condition of the matrix  Abar,
    # and the norms of  rbar  and  Abar'rbar.
    it.Acond =   it.Anorm * sqrt(it.ddnorm)
    res1     =   abs2(it.phibar)
    it.res2  =   it.res2 + abs2(psi)
    rnorm    =   sqrt(res1 + it.res2)
    Arnorm   =   it.alpha * abs(tau)

    # 07 Aug 2002:
    # Distinguish between
    #    r1norm = ||b - Ax|| and
    #    r2norm = rnorm in current code
    #           = sqrt(r1norm^2 + damp^2*||x||^2).
    #    Estimate r1norm from
    #    r1norm = sqrt(r2norm^2 - damp^2*||x||^2).
    # Although there is cancellation, it might be accurate enough.
    r1sq    =   abs2(rnorm) - it.dampsq * it.xxnorm
    r1norm  =   sqrt(abs(r1sq)); if r1sq < 0 r1norm = - r1norm; end
    r2norm  =   rnorm

    # Now use these norms to estimate certain other quantities,
    # some of which will be small near a solution.
    test1   =   rnorm / it.bnorm
    test2   =   Arnorm / (it.Anorm * rnorm)
    test3   =   inv(it.Acond)
    t1      =   test1 / (1 + it.Anorm * xnorm / it.bnorm)
    rtol    =   it.btol + it.atol * it.Anorm * xnorm / it.bnorm

    # The following tests guard against extremely small values of
    # atol, btol  or  ctol.  (The user may have set any or all of
    # the parameters  atol, btol, conlim  to 0.)
    # The effect is equivalent to the normal tests using
    # atol = eps,  btol = eps,  conlim = 1/eps.
    itn = iteration + 1
    istop = it.istop
    maxiter = it.maxiter
    atol = it.atol
    btol = it.btol
    ctol = it.ctol

    if itn >= maxiter  istop = 7; end
    if 1 + test3  <= 1 istop = 6; end
    if 1 + test2  <= 1 istop = 5; end
    if 1 + t1     <= 1 istop = 4; end

    # Allow for tolerances set by the user
    if  test3 <= ctol  istop = 3; end
    if  test2 <= atol  istop = 2; end
    if  test1 <= rtol  istop = 1; end

    it.istop = istop

    return (r1norm, test1, test2, test3), itn
end

function lsqr_iterator!(x, A, b;
    damp=0,
    atol=sqrt(eps(real(Adivtype(A, b)))),
    btol=sqrt(eps(real(Adivtype(A, b)))),
    conlim=real(one(Adivtype(A, b)))/sqrt(eps(real(Adivtype(A ,b)))),
    maxiter::Int=maximum(size(A))
    )
    #
    T  = Adivtype(A, b)
    Tr = real(T)

    ctol = conlim > 0 ? convert(Tr, 1/conlim) : zero(Tr)

    m, n = size(A)
    u = similar(b, T, m)
    v = similar(x, T, n)
    tmpm = similar(b, T, m)
    tmpn = similar(x, T, n)
    w = similar(v)
    wrho = similar(v)

    dampsq = abs2(damp)

    # Construct iterator and initialize based on RHS.
    # reset_rhs! will correctly set scalar fields.
    it = LSQRIterable(
        # LHS
        A, adjoint(A), x, damp, dampsq,
        # Bidiagonalization
        one(Tr), one(Tr), u, v,
        # Orthogonalization
        one(Tr), one(Tr), one(Tr), one(Tr),
        # Forward substitution
        w, wrho,
        # Stopping criterion
        one(Tr), one(Tr), one(Tr), one(Tr), one(Tr), one(Tr),
        # Worker arrays
        tmpm, tmpn,
        # Bookkeeping
        0, 0, maxiter, zero(Tr), atol, btol, ctol, 0
    )
    reset_rhs!(it, b)

    return it
end

"""
    lsqr(A, b; kwrags...) -> x, [history]

Same as [`lsqr!`](@ref), but allocates a solution vector `x` initialized with zeros.
"""
lsqr(A, b; kwargs...) = lsqr!(zerox(A, b), A, b; kwargs...)

"""
    lsqr!(x, A, b; kwargs...) -> x, [history]

Minimizes ``\\|Ax - b\\|^2 + \\|damp*x\\|^2`` in the Euclidean norm. If multiple solutions
exists returns the minimal norm solution.

The method is based on the Golub-Kahan bidiagonalization process.
It is algebraically equivalent to applying CG to the normal equations
``(A^*A + λ^2I)x = A^*b`` but has better numerical properties,
especially if A is ill-conditioned.

# Arguments
- `x`: Initial guess, will be updated in-place;
- `A`: linear operator;
- `b`: right-hand side.

## Keywords

- `damp::Number = 0`: damping parameter.
- `atol::Number = 1e-6`, `btol::Number = 1e-6`: stopping tolerances. If both are
  1.0e-9 (say), the final residual norm should be accurate to about 9 digits.
  (The final `x` will usually have fewer correct digits,
  depending on `cond(A)` and the size of damp).
- `conlim::Number = 1e8`: stopping tolerance.  `lsmr` terminates if an estimate
  of `cond(A)` exceeds conlim.  For compatible systems Ax = b,
  conlim could be as large as 1.0e+12 (say).  For least-squares
  problems, conlim should be less than 1.0e+8.
  Maximum precision can be obtained by setting
  `atol` = `btol` = `conlim` = zero, but the number of iterations
  may then be excessive.
- `maxiter::Int = maximum(size(A))`: maximum number of iterations.
- `verbose::Bool = false`: print method information.
- `log::Bool = false`: output an extra element of type `ConvergenceHistory`
  containing extra information of the method execution.

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
function lsqr!(x, A, b;
    maxiter::Int=maximum(size(A)), log::Bool=false, verbose::Bool=false,
    kwargs...
    )
    # Sanity-checking
    m = size(A, 1)
    n = size(A, 2)
    length(x) == n || error("x should be of length ", n)
    length(b) == m || error("b should be of length ", m)
    for i = 1:n
        isfinite(x[i]) || error("Initial guess for x must be finite")
    end

    history = ConvergenceHistory(partial=!log)
    log && reserve!(history, [:resnorm, :anorm, :rnorm, :cnorm], maxiter)

    # Run LSQR
    iterable = lsqr_iterator!(x, A, b; maxiter=maxiter, kwargs...)

    history[:atol] = iterable.atol
    history[:btol] = iterable.btol
    history[:ctol] = iterable.ctol

    verbose && @printf("=== lsqr ===\n%4s\t%7s\t\t%7s\t\t%7s\t\t%7s\n","iter","resnorm","anorm","cnorm","rnorm")

    for (iteration, item) = enumerate(iterable)
        # resnorm = r1norm = item[1]
        # rnorm   = test1  = item[2]
        # anorm   = test2  = item[3]
        # cnorm   = test3  = item[4]
        if log
            nextiter!(history, mvps = 1, mtvps = 1)
            push!(history, :resnorm, item[1])
            push!(history, :rnorm,   item[2])
            push!(history, :anorm,   item[3])
            push!(history, :cnorm,   item[4])
        end
        if verbose
            @printf("%3d\t%1.2e\t%1.2e\t%1.2e\t%1.2e\n",
                iteration, item[1], item[3], item[4], item[2])
        end
    end

    verbose && pritnln()
    log && setconv(history, converged(iterable))
    log && shrink!(history)

    log ? (iterable.x, history) : iterable.x
end

#########################
# Method Implementation #
#########################

# Michael Saunders, Systems Optimization Laboratory,
# Dept of MS&E, Stanford University.
#
# Adapted for Julia by Timothy E. Holy with the following changes:
#    - Allow an initial guess for x
#    - Eliminate printing
#----------------------------------------------------------------------
function lsqr_method!(log::ConvergenceHistory, x, A, b;
    damp=0, atol=sqrt(eps(real(Adivtype(A,b)))), btol=sqrt(eps(real(Adivtype(A,b)))),
    conlim=real(one(Adivtype(A,b)))/sqrt(eps(real(Adivtype(A,b)))),
    maxiter::Int=maximum(size(A)), verbose::Bool=false,
    )
    verbose && @printf("=== lsqr ===\n%4s\t%7s\t\t%7s\t\t%7s\t\t%7s\n","iter","resnorm","anorm","cnorm","rnorm")
    # Sanity-checking
    m = size(A,1)
    n = size(A,2)
    length(x) == n || error("x should be of length ", n)
    length(b) == m || error("b should be of length ", m)
    for i = 1:n
        isfinite(x[i]) || error("Initial guess for x must be finite")
    end

    # Initialize
    T = Adivtype(A, b)
    Tr = real(T)
    itn = istop = 0
    ctol = conlim > 0 ? convert(Tr, 1/conlim) : zero(Tr)
    Anorm = Acond = ddnorm = res2 = xnorm = xxnorm = z = sn2 = zero(Tr)
    cs2 = -one(Tr)
    dampsq = abs2(damp)
    tmpm = similar(b, T, m)
    tmpn = similar(x, T, n)

    log[:atol] = atol
    log[:btol] = btol
    log[:ctol] = ctol

    # Set up the first vectors u and v for the bidiagonalization.
    # These satisfy  beta*u = b-A*x,  alpha*v = A'u.
    u = b - A*x
    v = copy(x)
    beta = norm(u)
    alpha = zero(Tr)
    adjointA = adjoint(A)
    if beta > 0
        log.mtvps=1
        u .*= inv(beta)
        mul!(v, adjointA, u)
        alpha = norm(v)
    end
    if alpha > 0
        v .*= inv(alpha)
    end
    w = copy(v)
    wrho = similar(w)

    Arnorm = alpha*beta
    if Arnorm == 0
        return
    end

    rhobar = alpha
    phibar = bnorm = rnorm = r1norm = r2norm = beta

    #------------------------------------------------------------------
    #     Main iteration loop.
    #------------------------------------------------------------------
    while (itn < maxiter) & !log.isconverged
        nextiter!(log,mvps=1)
        itn += 1

        # Perform the next step of the bidiagonalization to obtain the
        # next beta, u, alpha, v.  These satisfy the relations
        #      beta*u  =  A*v  - alpha*u,
        #      alpha*v  =  A'*u - beta*v.

        # Note that the following three lines are a band aid for a GEMM: X: C := αAB + βC.
        # This is already supported in mul! for sparse and distributed matrices, but not yet dense
        mul!(tmpm, A, v)
        u .= -alpha .* u .+ tmpm
        beta = norm(u)
        if beta > 0
            log.mtvps+=1
            u .*= inv(beta)
            Anorm = sqrt(abs2(Anorm) + abs2(alpha) + abs2(beta) + dampsq)
            # Note that the following three lines are a band aid for a GEMM: X: C := αA'B + βC.
            # This is already supported in mul! for sparse and distributed matrices, but not yet dense
            mul!(tmpn, adjointA, u)
            v .= -beta .* v .+ tmpn
            alpha  = norm(v)
            if alpha > 0
                v .*= inv(alpha)
            end
        end

        # Use a plane rotation to eliminate the damping parameter.
        # This alters the diagonal (rhobar) of the lower-bidiagonal matrix.
        rhobar1 = sqrt(abs2(rhobar) + dampsq)
        cs1     = rhobar/rhobar1
        sn1     = damp  /rhobar1
        psi     = sn1*phibar
        phibar  = cs1*phibar

        # Use a plane rotation to eliminate the subdiagonal element (beta)
        # of the lower-bidiagonal matrix, giving an upper-bidiagonal matrix.
        rho     =   sqrt(abs2(rhobar1) + abs2(beta))
        cs      =   rhobar1/rho
        sn      =   beta   /rho
        theta   =   sn*alpha
        rhobar  = - cs*alpha
        phi     =   cs*phibar
        phibar  =   sn*phibar
        tau     =   sn*phi

        # Update x and w
        t1      =   phi  /rho
        t2      = - theta/rho

        x .+= t1*w
        w = t2 .* w .+ v
        wrho .= w .* inv(rho)
        ddnorm += norm(wrho)

        # Use a plane rotation on the right to eliminate the
        # super-diagonal element (theta) of the upper-bidiagonal matrix.
        # Then use the result to estimate  norm(x).
        delta   =   sn2*rho
        gambar  =  -cs2*rho
        rhs     =   phi - delta*z
        zbar    =   rhs/gambar
        xnorm   =   sqrt(xxnorm + abs2(zbar))
        gamma   =   sqrt(abs2(gambar) + abs2(theta))
        cs2     =   gambar/gamma
        sn2     =   theta /gamma
        z       =   rhs   /gamma
        xxnorm +=   abs2(z)

        # Test for convergence.
        # First, estimate the condition of the matrix  Abar,
        # and the norms of  rbar  and  Abar'rbar.
        Acond   =   Anorm*sqrt(ddnorm)
        res1    =   abs2(phibar)
        res2    =   res2 + abs2(psi)
        rnorm   =   sqrt(res1 + res2)
        Arnorm  =   alpha*abs(tau)

        # 07 Aug 2002:
        # Distinguish between
        #    r1norm = ||b - Ax|| and
        #    r2norm = rnorm in current code
        #           = sqrt(r1norm^2 + damp^2*||x||^2).
        #    Estimate r1norm from
        #    r1norm = sqrt(r2norm^2 - damp^2*||x||^2).
        # Although there is cancellation, it might be accurate enough.
        r1sq    =   abs2(rnorm) - dampsq*xxnorm
        r1norm  =   sqrt(abs(r1sq));   if r1sq < 0 r1norm = - r1norm; end
        r2norm  =   rnorm
        push!(log, :resnorm, r1norm)

        # Now use these norms to estimate certain other quantities,
        # some of which will be small near a solution.
        test1   =   rnorm /bnorm
        test2   =   Arnorm/(Anorm*rnorm)
        test3   =   inv(Acond)
        t1      =   test1/(1 + Anorm*xnorm/bnorm)
        rtol    =   btol + atol*Anorm*xnorm/bnorm
        push!(log, :cnorm, test3)
        push!(log, :anorm, test2)
        push!(log, :rnorm, test1)
        verbose && @printf("%3d\t%1.2e\t%1.2e\t%1.2e\t%1.2e\n",itn,r1norm,test2,test3,test1)

        # The following tests guard against extremely small values of
        # atol, btol  or  ctol.  (The user may have set any or all of
        # the parameters  atol, btol, conlim  to 0.)
        # The effect is equivalent to the normal tests using
        # atol = eps,  btol = eps,  conlim = 1/eps.
        if itn >= maxiter  istop = 7; end
        if 1 + test3  <= 1 istop = 6; end
        if 1 + test2  <= 1 istop = 5; end
        if 1 + t1     <= 1 istop = 4; end

        # Allow for tolerances set by the user
        if  test3 <= ctol  istop = 3; end
        if  test2 <= atol  istop = 2; end
        if  test1 <= rtol  istop = 1; end

        setconv(log, istop > 0)
    end
    verbose && @printf("\n")
    x
end
