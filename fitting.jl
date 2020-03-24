using Flux;
using Flux: train!, throttle;
using LinearAlgebra;
using SparseArrays;
using LightGraphs;
using Plots; pyplot();
using LaTeXStrings;
using Distributions;
using Einsum;

include("utils.jl");
include("kernels.jl");

G = watts_strogatz(10, 4, 0.3);

p1 = 3;
p2, s, d = 2, [2,3], [2,2];
t, k = 128, 32;

p = p1 + sum(d);
n = nv(G);
A = getA(G, p);
D = A2D.(A);
N = 1000;

FIDX(fidx, V=vertices(G)) = [(i-1)*p+j for i in V for j in fidx];
ss = p1 .+ vcat(0, cumsum(d)[1:end-1]) .+ 1;
ff = p1 .+ cumsum(d);
cr = [ss_:ff_ for (ss_,ff_) in zip(ss,ff)];

α0 = vcat(randn(p), randn(div(p*(p-1),2)));
β0 = exp(randn());
CM0 = inv(Array(getΓ(α0, β0; A=A)));
CM = (CM0 + CM0')/2.0;
g = MvNormal(CM);
YZ = cat([reshape(rand(g), (p,n)) for _ in 1:N]..., dims=3);
Y = YZ[1:p1,:,:];
Z = [YZ[cr_,:,:] for cr_ in cr];

einsum0(A, B) = @einsum t[i,k,l] := A[i,j] * B[j,k,l];

W0 = [randn(s[j], d[j]) for j in 1:p2];
PX = [softmax(einsum0(W0_,Z_), dims=1) for (W0_,Z_) in zip(W0,Z)];

function sample_replace(px)
    x = zeros(size(px));
    for j in 1:size(px,2)
        for k in 1:size(px,3)
            x[sample(Weights(px[:,j,k])),j,k] = 1;
        end
    end
    return x;
end
X = sample_replace.(PX);

φ = param(zeros(div(p*(p+1),2)+1));
W = [param(w0) for w0 in W0];
μ = [param(randn(d[j], s[j])) for j in 1:p2];
logσ = [param(randn(d[j], s[j])) for j in 1:p2];

getα() = vcat(φ[1:p], φ[p+1:end-1]);
getβ() = exp(φ[end]);
getμ(X) = reshape(cat([einsum0(μ_, X_) for (μ_, X_) in zip(μ, X)]..., dims=1), (sum(d)*n, N));
getσ(X) = reshape(cat([exp.(einsum0(logσ_, X_)) for (logσ_, X_) in zip(logσ, X)]..., dims=1), (sum(d)*n, N));

print_params() = @printf("α:  %s,    β:  %10.3f\n", array2str(getα()), getβ());

function logPY(Y; getα=getα, getβ=getβ)
    yy = []
    rr = [y for y in yy];
    V = collect(1:size(A[1],1));
    L = FIDX(1:p1);
    U = setdiff(V,L);

    # Γ = Tracker.collect(getΓ(getα(), getβ(); A=A));
    # ΓSC = Γ[L,L] - Γ[L,U] * inv(Γ[U,U]) * Γ[U,L];
    # Ω = mean([r' * ΓSC * r for r in rr]) - logdet(ΓSC);

    Ω = mean([quadformSC(getα(), getβ(), r; A=A, L=L) for r in rr]) - logdetΓ(getα(), getβ(); A=A, P=V, t=t, k=k) + logdetΓ(getα(), getβ(); A=A, P=U, t=t, k=k);
    Ω /= length(L);
    # Ω += 0.1 * norm(getα()*getβ(), 1);

    return Ω;
end

getL(l::AbstractVector) = vcat([collect((i-1)*p+1:i*p) for i in l]...);
getL(u::Int) = getL(vcat(neighbors(G,u), u));

dat = [(L->([sample(X)[L] for _ in 1:1],L))(getL(1:n)) for _ in 1:10000];

# train!(loss, Flux.params(φ), dat, Descent(0.01), cb = throttle(print_params, 10));


#----------------------------------------------------
# Analysis
#----------------------------------------------------
# α1 = CSV.read("output", delim=" ", ignorerepeated=true, header=0)[end,2:div(p*(p+1),2)+1] |> collect |> xx->[parse(Float64,x[1:end-1]) for x in xx];
# β1 = CSV.read("output", delim=" ", ignorerepeated=true, header=0)[end,end];
# FIDX(fidx, V=vertices(G)) = [(i-1)*p+j for i in V for j in fidx];
# L, U = rand_split(nv(G), 0.6);
# 
# function print_vol(lidx=[7], fidx=[1,2,3,4,5,6])
#     obsl = FIDX(lidx, L); obsf = FIDX(fidx, vertices(G)); trgl = FIDX(lidx, U);
#     vol_base() = (-logdetΓ(param(α1), param(β1); A=A, P=vcat(obsf,obsl,trgl), t=t, k=k)) - (-logdetΓ(param(α1), param(β1); A=A, P=vcat(obsf,obsl), t=t, k=k));
#     vol_lp()  = (-logdetΓ(param(α1), param(β1); A=A, P=vcat(obsf,trgl), t=t, k=k)) - (-logdetΓ(param(α1), param(β1); A=A, P=obsf, t=t, k=k));
#     vol_gnn() = (-logdetΓ(param(α1), param(β1); A=A, P=vcat(obsl,trgl), t=t, k=k)) - (-logdetΓ(param(α1), param(β1); A=A, P=obsl, t=t, k=k));
#     vol_cgnn() = (-logdetΓ(param(α1), param(β1); A=A, P=trgl, t=t, k=k));
# 
#     @printf("base:    %10.4f\n", mean([vol_base() for _ in 1:10]));
#     @printf("lp:      %10.4f\n", mean([vol_lp() for _ in 1:10]));
#     @printf("gnn:     %10.4f\n", mean([vol_gnn() for _ in 1:10]));
#     @printf("cgnn:    %10.4f\n", mean([vol_cgnn() for _ in 1:10]));
# end
