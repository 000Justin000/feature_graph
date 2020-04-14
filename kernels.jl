using Random;
using StatsBase;
using LightGraphs;
using LinearAlgebra
using SparseArrays;
using Arpack;
using Flux;
using Flux.Tracker: data, track, @grad, forward, Params, update!, back!, grad, gradient;
using Flux.Tracker: TrackedReal, TrackedVector, TrackedMatrix;
using IterativeSolvers;
using MLBase: roc, f1score;
using Printf;
using PyCall;

include("utils.jl");

function mBCG(mmm_A::Function, B::Array{Float64,2}; PC::Function=Y->Y, k::Int=size(B,1), tol=1.0e-6)
    """
    Args:
     mmm_A: matrix matrix multiplication routine
         B: right-hand-side vectors
        PC: apply preconditioner to each column of the matrix
         k: max # of iterations
       tol: error tolerance

    Returns:
         X: solution vectors
        TT: Lanczos tridiagonal matrices
    """

    n,t = size(B);
    X = zeros(n,t);
    R = B - mmm_A(X);
    Z = PC(R);
    P = Z;
    α = zeros(t);
    β = zeros(t);

    T = [(dv=Vector{Float64}(), ev=Vector{Float64}()) for _ in 1:t];

    tol_vec = tol .+ tol*sqrt.(sum(R.*R, dims=1)[:]);
    for j in 1:k
        if all(sqrt.(sum(R.*R, dims=1)[:]) .< tol_vec)
            break;
        end

        AP = mmm_A(P);
        α_ = sum(R.*Z, dims=1)[:] ./ sum(P.*AP, dims=1)[:];
        X_ = X + P .* α_';
        R_ = R - AP .* α_';
        Z_ = PC(R_);
        β_ = sum(Z_.*R_, dims=1)[:] ./ sum(Z.*R, dims=1)[:];
        P_ = Z_ + P .* β_';

        for i in 1:t
            if j == 1
                push!(T[i].dv, 1.0/α_[i]);
            else
                push!(T[i].dv, 1.0/α_[i]+β[i]/α[i]);
                push!(T[i].ev, sqrt(β[i])/α[i]);
            end
        end

        P = P_;
        R = R_;
        Z = Z_;
        X = X_;
        α = α_;
        β = β_;
    end

    return X, [SymTridiagonal(dv,ev) for (dv,ev) in T];
end

function getΓ(α, β; A)
    return β * I + β * sum([(abs(α_)*D_ - α_*A_) for (α_,D_,A_) in zip(α,A2D.(A),A)]);
end

function get∂Γ∂α(α, β; A)
    return [β*sign(α_)*D_ - β*A_ for (α_,D_,A_) in zip(α,A2D.(A),A)];
end

function get∂Γ∂β(α, β; A)
    return I + sum([(abs(α_)*D_ - α_*A_) for (α_,D_,A_) in zip(α,A2D.(A),A)]);
end

logdetΓ(α::TrackedVector, β::TrackedReal; A, P, t, k) = track(logdetΓ, α, β; A=A, P=P, t=t, k=k);
@grad function logdetΓ(α, β; A, P, t, k)
    """
    Args:
         α: model parameter vector
         β: model parameter
         A: adjacency matrix vector
         P: index set
         t: # of trial vectors
         k: # of Lanczos tridiagonal iterations

    Return:
         log determinant of the principle submatrix ΓPP
    """

    (length(P) == 0) && return 0.0, Δ -> (zeros(length(α)), 0.0);

    α = data(α);
    β = data(β);

    n = length(P);
    Z = randn(n,t);

    Γ = getΓ(α, β; A=A);
    ∂Γ∂α = get∂Γ∂α(α, β; A=A);
    ∂Γ∂β = get∂Γ∂β(α, β; A=A);

    X, TT = mBCG(Y->Γ[P,P]*Y, Z; k=k);

    vv = 0;
    for T in TT
        eigvals, eigvecs = eigen(T);
        vv += sum(eigvecs[1,:].^2 .* log.(eigvals));
    end

    Ω = vv*n/t;

    trΓiM(M) = sum(X.*(M[P,P]*Z))/t;
    ∂Ω∂α = map(trΓiM, ∂Γ∂α);
    ∂Ω∂β = trΓiM(∂Γ∂β);

    return Ω, Δ -> (Δ*∂Ω∂α, Δ*∂Ω∂β)
end

function test_logdetΓ(n=100)
    G = random_regular_graph(n, 3);
    A = [adjacency_matrix(G)];
    L = randperm(n)[1:div(n,2)];

    #------------------------
    p = param(randn(2));
    getα() = p[1:1];
    getβ() = softplus(p[2]);
    #------------------------

    #------------------------
    # true value
    #------------------------
    Γ = Tracker.collect(getΓ(getα(), getβ(); A=A));
    Ω = logdet(Tracker.collect(Γ[L,L]));
    #------------------------
    Tracker.back!(Ω, 1);
    @printf("accurate:       [%s]\n", array2str(Tracker.grad(p)));
    reset_grad!(p);
    #------------------------

    #------------------------
    # approximation
    #------------------------
    Ω = logdetΓ(getα(), getβ(); A=A, P=L, t=128, k=32);
    #------------------------
    Tracker.back!(Ω, 1);
    @printf("approximate:    [%s]\n", array2str(Tracker.grad(p)));
    reset_grad!(p);
    #------------------------
end

quadformSC(α::TrackedVector, β::TrackedReal, rL; A, L) = track(quadformSC, α, β, rL; A=A, L=L);
@grad function quadformSC(α, β, rL; A, L)
    """
    Args:
         α: model parameter vector
         β: model parameter
        rL: noise on vertex set L
         A: adjacency matrix vector
         L: index set

    Return:
         quadratic form: rL' (ΓLL - ΓLU ΓUU^-1 ΓUL) rL
    """

    α = data(α);
    β = data(β);
    rL = data(rL);

    Γ = getΓ(α, β; A=A);
    ∂Γ∂α = get∂Γ∂α(α, β; A=A);
    ∂Γ∂β = get∂Γ∂β(α, β; A=A);

    U = setdiff(1:size(A[1],1), L);

    Ω = rL'*Γ[L,L]*rL - rL'*Γ[L,U]*cg(Γ[U,U],Γ[U,L]*rL);

    quadform_partials(M) = rL'*M[L,L]*rL - rL'*M[L,U]*cg(Γ[U,U],Γ[U,L]*rL) + rL'*Γ[L,U]*cg(Γ[U,U],M[U,U]*cg(Γ[U,U],Γ[U,L]*rL)) - rL'*Γ[L,U]*cg(Γ[U,U],M[U,L]*rL);
    ∂Ω∂α = map(quadform_partials, ∂Γ∂α);
    ∂Ω∂β = quadform_partials(∂Γ∂β);
    ∂Ω∂rL = 2*Γ[L,L]*rL - 2*Γ[L,U]*cg(Γ[U,U],Γ[U,L]*rL);

    return Ω, Δ -> (Δ*∂Ω∂α, Δ*∂Ω∂β, Δ*∂Ω∂rL);
end

function test_quadformSC(n=100)
    G = random_regular_graph(n, 3);
    A = [adjacency_matrix(G)];

    #------------------------
    L = randperm(n)[1:div(n,2)];
    U = setdiff(1:n, L);
    rL = param(randn(div(n,2)));
    getrL() = rL[:];
    #------------------------

    #------------------------
    p = param(randn(2));
    getα() = p[1:1];
    getβ() = softplus(p[2]);
    #------------------------

    #------------------------
    # true value
    #------------------------
    Γ = Tracker.collect(getΓ(getα(), getβ(); A=A));
    SC = Γ[L,L] - Γ[L,U]*inv(Γ[U,U])*Γ[U,L];
    Ω = getrL()' * SC * getrL();
    #------------------------
    Tracker.back!(Ω, 1);
    @printf("accurate:       [%s],    [%s]\n", array2str(Tracker.grad(p)), array2str(Tracker.grad(rL)[1:10]));
    reset_grad!(p, rL);
    #------------------------

    #------------------------
    # approximation
    #------------------------
    Ω = quadformSC(getα(), getβ(), getrL(); A=A, L=L);
    #------------------------
    Tracker.back!(Ω, 1);
    @printf("accurate:       [%s],    [%s]\n", array2str(Tracker.grad(p)), array2str(Tracker.grad(rL)[1:10]));
    reset_grad!(p, rL);
    #------------------------
end

traceΓB(α::TrackedVector, β::TrackedReal, B; A, P) = track(traceΓB, α, β, B; A=A, P=P);
@grad function traceΓB(α, β, B; A, P)
    """
    Args:
         α: model parameter vector
         β: model parameter
         B: diagonal blocks of a matrix
         A: adjacency matrix vector
         P: index set

    Return:
         tr(ΓPP * blockdiag(B[:,:,i] for i in 1:size(B,3)))
    """
    @assert (size(B,1) == size(B,2)) && (length(P) == size(B,2) * size(B,3));

    α = data(α);
    β = data(β);
    B = data(B);

    Γ = getΓ(α, β; A=A);
    ∂Γ∂α = get∂Γ∂α(α, β; A=A);
    ∂Γ∂β = get∂Γ∂β(α, β; A=A);

    U = setdiff(1:size(A[1],1), L);

    Ω = rL'*Γ[L,L]*rL - rL'*Γ[L,U]*cg(Γ[U,U],Γ[U,L]*rL);

    quadform_partials(M) = rL'*M[L,L]*rL - rL'*M[L,U]*cg(Γ[U,U],Γ[U,L]*rL) + rL'*Γ[L,U]*cg(Γ[U,U],M[U,U]*cg(Γ[U,U],Γ[U,L]*rL)) - rL'*Γ[L,U]*cg(Γ[U,U],M[U,L]*rL);
    ∂Ω∂α = map(quadform_partials, ∂Γ∂α);
    ∂Ω∂β = quadform_partials(∂Γ∂β);
    ∂Ω∂rL = 2*Γ[L,L]*rL - 2*Γ[L,U]*cg(Γ[U,U],Γ[U,L]*rL);

    return Ω, Δ -> (Δ*∂Ω∂α, Δ*∂Ω∂β, Δ*∂Ω∂rL);
end

function test_traceΓB(n=100)
    G = random_regular_graph(n, 3);
    A = [adjacency_matrix(G)];

    #------------------------
    L = randperm(n)[1:div(n,2)];
    U = setdiff(1:n, L);
    rL = param(randn(div(n,2)));
    getrL() = rL[:];
    #------------------------

    #------------------------
    p = param(randn(2));
    getα() = p[1:1];
    getβ() = softplus(p[2]);
    #------------------------

    #------------------------
    # true value
    #------------------------
    Γ = Tracker.collect(getΓ(getα(), getβ(); A=A));
    SC = Γ[L,L] - Γ[L,U]*inv(Γ[U,U])*Γ[U,L];
    Ω = getrL()' * SC * getrL();
    #------------------------
    Tracker.back!(Ω, 1);
    @printf("accurate:       [%s],    [%s]\n", array2str(Tracker.grad(p)), array2str(Tracker.grad(rL)[1:10]));
    reset_grad!(p, rL);
    #------------------------

    #------------------------
    # approximation
    #------------------------
    Ω = quadformSC(getα(), getβ(), getrL(); A=A, L=L);
    #------------------------
    Tracker.back!(Ω, 1);
    @printf("accurate:       [%s],    [%s]\n", array2str(Tracker.grad(p)), array2str(Tracker.grad(rL)[1:10]));
    reset_grad!(p, rL);
    #------------------------
end

chol(A::TrackedArray) = track(chol, A);
@grad function chol(A)
    A = data(A);
    CF = cholesky(A);
    L, U = CF.L, CF.U;

    Φ(A) = LowerTriangular(A) - 0.5 * Diagonal(A);

    function sensitivity(ΔL)
        S = inv(U) * Φ(U * LowerTriangular(ΔL)) * inv(L);
        # return tuple(Matrix(S + S' - Diagonal(S)));
        return tuple(Matrix(0.5 * (S + S')));
    end

    return Matrix(L), sensitivity;
end

function test_chol(n=10)
    X = randn(n,n);
    A0 = X * X';
    A = param(A0);
    b = randn(n);

    Ω(A, b) = b' * A * b;

    reset_grad!(A);
    Tracker.back!(Ω(chol(A), b), 1);
    @printf("reverse-mode automatic differentiation\n");
    display(A.grad);
    @printf("\n\n");

    B = Matrix{eltype(A)}(undef, n, n);
    B .= A;
    reset_grad!(A);
    Tracker.back!(Ω(cholesky(B).L, b), 1);
    @printf("elementwise automatic differentation\n");
    display(A.grad);
    @printf("\n\n");

    ϵ = 1.0e-6;
    sen = zeros(n,n);
    for i in 1:n
        for j in 1:n
            Ap, Am = Array(A0), Array(A0);
            Ap[i,j] += ϵ;
            Ap[j,i] += ϵ;
            Am[i,j] -= ϵ    ;
            Am[j,i] -= ϵ;

            sen[i,j] = (Ω(cholesky(Ap).L, b) - Ω(cholesky(Am).L, b)) / (4 * ϵ);
            sen[j,i] = (Ω(cholesky(Ap).L, b) - Ω(cholesky(Am).L, b)) / (4 * ϵ);
        end
    end

    @printf("finite-difference\n");
    display(sen);
end
