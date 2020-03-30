using Flux;
using Flux: train!, throttle, Tracker;
using LinearAlgebra;
using SparseArrays;
using LightGraphs;
using Plots; pyplot();
using LaTeXStrings;
using Distributions;

include("utils.jl");
include("kernels.jl");

G = watts_strogatz(10, 4, 0.3);

p1 = 3;
p2, s, d = 1, [2], [2];
t, k = 128, 32;

p = p1 + sum(d);
n = nv(G);
A = getA(G, p);
D = A2D.(A);
N = 1000;
n_batch = 32;

FIDX(fidx, V=vertices(G)) = [(i-1)*p+j for i in V for j in fidx];
ss = vcat(0, cumsum(d)[1:end-1]) .+ 1;
ff = cumsum(d);
cr = [p1+ss_:p1+ff_ for (ss_,ff_) in zip(ss,ff)];

V = collect(1:size(A[1],1));
L = FIDX(1:p1);
U = setdiff(V,L);

# α0 = vcat(ones(p), ones(div(p*(p-1),2))*5.0);
α0 = vcat(randn(p), randn(div(p*(p-1),2)));
β0 = 1.0
# β0 = exp(randn());
CM0 = inv(Array(getΓ(α0, β0; A=A)));
CM = (CM0 + CM0')/2.0;
g = MvNormal(CM);
YZ = cat([reshape(rand(g), (p,n)) for _ in 1:N]..., dims=3);
Y = YZ[1:p1,:,:];
Z = [YZ[cr_,:,:] for cr_ in cr];

@printf("α0: %s,    β0: %10.3f\n", array2str(α0), β0);

tsctc(A, B) = reshape(A * reshape(B, (size(B,1), :)), (size(A,1), size(B)[2:end]...));

# W0 = [[-5.0, 5.0] for j in 1:p2];
W0 = [diagm(0=>ones(s[j])*5.0) for j in 1:p2];
# W0 = [randn(s[j], d[j]) for j in 1:p2];

b0 = [randn(s[j]) for j in 1:p2];

φ = param(zeros(div(p*(p+1),2)+1));
W = [param(W_) for W_ in W0];
b = [param(b_) for b_ in b0];
μ = [param(randn(d[j], s[j])) for j in 1:p2];
logσ = [param(randn(d[j], s[j])) for j in 1:p2];

logPX(Z) = [logsoftmax(tsctc(W_,Z_) .+ repeat(b_,1,n,size(Z_,3)), dims=1) for (W_,Z_,b_) in zip(W,Z,b)];

function sample_from(logpx)
    x = zeros(size(logpx));
    for j in 1:size(logpx,2)
        for k in 1:size(logpx,3)
            x[sample(Weights(exp.(logpx[:,j,k]))),j,k] = 1;
        end
    end
    return x;
end
X = sample_from.(logPX(Z));

getα() = vcat(φ[1:p], φ[p+1:end-1]);
getβ() = exp(φ[end]);

function Qzμσ0(X, Y)
    μZ0 = [tsctc(μ_, X_) for (μ_, X_) in zip(μ, X)];
    σZ0 = [exp.(tsctc(logσ_, X_)) for (logσ_, X_) in zip(logσ, X)];

    return μZ0, σZ0;
end

function Qzμσ1(X, Y)
    batch_size = size(Y,3);

    μZ0, σZ0 = Qzμσ0(X, Y);

    C1 = [σZ0_.^-2.0 .* μZ0_ for (μZ0_,σZ0_) in zip(μZ0,σZ0)];

    ΓUL = getΓ(getα(), getβ(); A=A)[U,L];
    C2S = reshape(-ΓUL * reshape(Y, (p1*n, batch_size)), (sum(d), n, batch_size));
    C2 = [Tracker.collect(C2S[ss_:ff_,:,:]) for (ss_,ff_) in zip(ss,ff)];

    σZ1 = [(σZ0_.^-2.0 .+ getβ()).^-0.5 for σZ0_ in σZ0];
    μZ1 = [σZ1_.^2.0 .* (C1_ .+ C2_) for (σZ1_, C1_, C2_) in zip(σZ1, C1, C2)];

    return μZ1, σZ1;
end

Qzμσ = Qzμσ1;

function EQzlogPX(X, Y)
    ZS = [μZ_ + σZ_ .* randn(size(σZ_)) for (μZ_, σZ_) in zip(Qzμσ(X, Y)...)];
    Ω = sum(sum(logPX_ .* X_) for (logPX_,X_) in zip(logPX(ZS),X));

    return Ω;
end

function loss(X, Y)
    batch_size = size(Y,3);

    μZ, σZ = Qzμσ(X, Y);

    μZS = cat(μZ..., dims=1);
    μzs = [vec(μZS[:,:,i]) for i in 1:size(μZS,3)];

    σZS = cat(σZ..., dims=1);
    σzs = [vec(σZS[:,:,i]) for i in 1:size(σZS,3)];

    YZS = cat(Y,μZS, dims=1);
    yzs = [vec(YZS[:,:,i]) for i in 1:size(YZS,3)];

    diagΓUU = Tracker.collect(diag(getΓ(getα(), getβ(); A=A)[U,U]));
    Ω = 0.5 * logdetΓ(getα(), getβ(); A=A, P=V, t=t, k=k);
    Ω -= 0.5 * mean(quadformSC(getα(), getβ(), yzs_; A=A, L=V) for yzs_ in yzs);
    Ω -= 0.5 * mean(dot(diagΓUU, σz.^2) for σz in σzs);
    Ω += sum(log.(σZS)) / batch_size;
    Ω += EQzlogPX(X, Y) / batch_size;

    return -Ω;
end

dat = [(L->([X_[:,:,L] for X_ in X], Y[:,:,L]))(sample(1:N, n_batch)) for _ in 1:50000];

print_params() = @printf("α:  %s,    β:  %10.3f\n", array2str(getα()), getβ());
train!(loss, Flux.params(φ, μ..., logσ...), dat, Descent(0.01), cb = throttle(print_params, 10));


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
