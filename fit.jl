using Flux;
using Flux: train!, throttle;
using LinearAlgebra;
using SparseArrays;
using LightGraphs;
using Plots; pyplot();
using LaTeXStrings;
using Distributions;

include("utils.jl");
include("kernels.jl");
include("read_network.jl");

t, k = 128, 32;
# p = 3; G = random_regular_graph(100, 5);
p = 1; G = random_regular_graph(100, 5);
n = nv(G);
A = getA(G, p);
D = A2D.(A);

α0 = vcat(randn(p), randn(div(p*(p-1),2)), randn(p));
β0 = exp(randn());
CM0 = inv(Array(getΓ(α0, β0; A=A)));
CM = (CM0 + CM0')/2.0;
g = MvNormal(CM);
X = [rand(g) for _ in 1:1];

φ = param(zeros(div(p*(p+3),2)+1));
getα() = vcat(φ[1:p], φ[p+1:end-1-p], φ[end-p:end-1]);
getβ() = exp(φ[end]);
print_params() = @printf("α:  %s,    β:  %10.3f\n", array2str(getα()), getβ());

function loss(xx, L; getα=getα, getβ=getβ)
    rr = [x for x in xx];

    U = setdiff(1:n*2*p,L);

    # Γ = Tracker.collect(getΓ(getα(), getβ(); A=A));
    # ΓSC = Γ[L,L] - Γ[L,U] * inv(Γ[U,U]) * Γ[U,L];
    # Ω = mean([r' * ΓSC * r for r in rr]) - logdet(ΓSC);

    Ω = mean([quadformSC(getα(), getβ(), r; A=A, L=L) for r in rr]) - logdetΓ(getα(), getβ(); A=A, P=collect(1:n*2*p), t=t, k=k) + logdetΓ(getα(), getβ(); A=A, P=U, t=t, k=k);

    return Ω;
end

getL(l::AbstractVector) = vcat([collect(i*2*p-p+1:i*2*p) for i in l]...);
getL(u::Int) = getL(vcat(neighbors(G,u), u));

dat = [(L->([sample(X)[L] for _ in 1:1],L))(getL(1:n)) for _ in 1:10000];

@printf("α0: %s,    β0: %10.3f\n", array2str(α0), β0);
train!(loss, Flux.params(φ), dat, Descent(0.001), cb = throttle(print_params, 10));
