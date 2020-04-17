using Flux;
using Flux: train!, throttle, Tracker;
using LinearAlgebra;
using SparseArrays;
using LightGraphs;
using Plots;
using LaTeXStrings;
using Distributions;
using FastGaussQuadrature;
using SpecialFunctions;
using Random;
using GraphSAGE;

include("utils.jl");
include("kernels.jl");

Random.seed!(0);

# G = complete_graph(2);
G = watts_strogatz(10, 4, 0.3);
# G = watts_strogatz(3000, 6, 0.3);

p1 = 2;
p2, s, d = 1, Int[2], Int[1];
t, k, glm, dim_h, dim_r = 128, 32, 100, 32, 8;

p = p1 + sum(d);
n = nv(G);
A = getA(G, p);
D = A2D.(A);
N = 1024;
n_batch = 32;

FIDX(fidx, V=vertices(G)) = [(i-1)*p+j for i in V for j in fidx];
ll = vcat(0, cumsum(d));
ss = [ll[i]+1 for i in 1:p2];
ff = [ll[i+1] for i in 1:p2];
cr = [p1+ss_:p1+ff_ for (ss_,ff_) in zip(ss,ff)];

V = collect(1:size(A[1],1));
L = FIDX(1:p1);
U = setdiff(V,L);

α0 = vcat(ones(p), rand(length(A)-p));
β0 = 1.0;
# α0 = vcat(randn(p), randn(div(p*(p-1),2)));
# β0 = exp(randn());
CM0 = inv(Array(getΓ(α0, β0; A=A)));
CM = (CM0 + CM0')/2.0;
g = MvNormal(CM);
YZ = cat([reshape(rand(g), (p,n)) for _ in 1:N]..., dims=3);
Y = YZ[1:p1,:,:];
Z = [YZ[cr_,:,:] for cr_ in cr];

@printf("α0: %s,    β0: %10.3f\n", array2str(α0), β0);

tsctc(A, B) = reshape(A * reshape(B, (size(B,1), :)), (size(A,1), size(B)[2:end]...));

# parameter used as decoder for synthetic data
W0 = [[-5.0, 5.0] for j in 1:p2];
# W0 = [diagm(0=>ones(s[j])*5.0) for j in 1:p2];
b0 = [zeros(s[j]) for j in 1:p2];

# parameter for the latent Gaussian (to be learned)
φ = param(zeros(length(A)+1));

# parameter for the decoder (fixed)
W = [param(W_) for W_ in W0];
b = [param(b_) for b_ in b0];

# parameter for the encoder (to be learned)
μ = [param(zeros(d[j], s[j])) for j in 1:p2];
logσ = [param(zeros(d[j], s[j])) for j in 1:p2];
η = [param(zeros(d[j], s[j])) for j in 1:p2];
enc = graph_encoder(p1+sum(s), dim_r, dim_h, repeat(["SAGE_Mean"], 2); σ=relu);
reg = Dense(dim_r, 2*sum(d));

logPX(Z) = [logsoftmax(tsctc(W_,Z_) .+ repeat(b_,1,n,size(Z_,3)), dims=1) for (W_,Z_,b_) in zip(W,Z,b)];

X = [begin
        x = zeros(size(logPX_));
        for j in 1:size(logPX_,2)
            for k in 1:size(logPX_,3)
                x[sample(Weights(exp.(logPX_[:,j,k]))),j,k] = 1;
            end
        end
        x;
     end for logPX_ in logPX(Z)];

getα() = φ[1:end-1];
getβ() = exp(φ[end]);
# getβ() = param(1.0);

# TO BE VERIFIED
# rho(σZ_, ηZ_) = (σZ_ .* σZ_ .* ηZ_) ./ sqrt.(1 .+ sum((σZ_ .* ηZ_) .* (σZ_ .* ηZ_), dims=1));
rho(σZ_, ηZ_) = (σZ_ .* ηZ_) ./ sqrt.(1 .+ sum(ηZ_ .* ηZ_, dims=1));

# return normal distribution with diagonal covariance matrix, conditioned on X
function Qzμσ0(X, Y)
    μZ0 = [tsctc(μ_, X_) for (μ_, X_) in zip(μ, X)];
    σZ0 = [exp.(tsctc(logσ_, X_)) for (logσ_, X_) in zip(logσ, X)];

    return μZ0, σZ0;
end

# return normal distribution with diagonal covariance matrix, conditioned on X & Y
function Qzμσ1(X, Y)
    batch_size = size(Y,3);

    μZ0, σZ0 = Qzμσ0(X, Y);

    C1 = [σZ0_.^-2.0 .* μZ0_ for (μZ0_,σZ0_) in zip(μZ0,σZ0)];

    C2S = reshape(-ΓX(getα(), getβ(), reshape(Y, (p1*n, batch_size)); A=A, U=U, L=L), (sum(d), n, batch_size));
    C2 = [C2S[ss_:ff_,:,:] for (ss_,ff_) in zip(ss,ff)];

    σZ1 = [(σZ0_.^-2.0 .+ getβ()).^-0.5 for σZ0_ in σZ0];
    μZ1 = [σZ1_.^2.0 .* (C1_ .+ C2_) for (σZ1_, C1_, C2_) in zip(σZ1, C1, C2)];

    return μZ1, σZ1;
end

# use GCN to learn μZ0 and σZ0
function Qzμσ2(X, Y)
    batch_size = size(Y,3);

    YX = cat(Y, X...; dims=1);
    μlogσZ = Flux.stack([reg(hcat(enc(G, collect(1:n), u->YX[:,u,i])...)) for i in 1:batch_size], 3);

    μZ, σZ = μlogσZ[1:sum(d),:,:], exp.(μlogσZ[sum(d)+1:end,:,:]);
    μZ2 = [μZ[ss_:ff_,:,:] for (ss_,ff_) in zip(ss,ff)];
    σZ2 = [σZ[ss_:ff_,:,:] for (ss_,ff_) in zip(ss,ff)];

    return μZ2, σZ2;
end

# sample normal distribution with diagonal covariance matrix

function sample_μσ(μZ, σZ)
    ZS = [μZ_ + σZ_ .* randn(size(σZ_)) for (μZ_, σZ_) in zip(μZ, σZ)];

    return ZS;
end

# return skewed normal distribution with diagonal covariance matrix, conditioned on X
function Qzμση0(X, Y)
    μZ0 = [tsctc(μ_, X_) for (μ_, X_) in zip(μ, X)];
    σZ0 = [exp.(tsctc(logσ_, X_)) for (logσ_, X_) in zip(logσ, X)];
    ηZ0 = [tsctc(η_, X_) for (η_, X_) in zip(η, X)];

    return μZ0, σZ0, ηZ0;
end

# sample skewed normal distribution with diagonal covariance matrix
function sample_μση(μZ, σZ, ηZ)
    ZS = [];
    for (μZ_, σZ_, ηZ_) in zip(μZ, σZ, ηZ)
        d_, batch_size, etype = size(μZ_,1), size(μZ_,3), eltype(μZ_);

        ρZ_ = rho(σZ_, ηZ_);
        ZS_ = Array{etype}(undef, size(μZ_)...);
        Threads.@threads for k in 1:batch_size
            for j in 1:n
                CM_ = diagm(0=>σZ_[:,j,k] .* σZ_[:,j,k]) - ρZ_[:,j,k] * ρZ_[:,j,k]';
                ZS_[:,j,k] .= μZ_[:,j,k] .+ ρZ_[:,j,k] * abs(randn()) + chol(CM_) * randn(d_);
            end
        end

        push!(ZS, Tracker.collect(ZS_));
    end

    return ZS;
end

function H_N(μ, σ)
    """
    Args:
        μ: center of the Gaussian
        σ: spread parameter

    Return:
        entropy of the Gaussian
    """
    @assert length(μ) == length(σ);
    k = length(μ);

    H0 = 0.5 * k + 0.5 * k * log(2π) + sum(log.(σ));

    return H0;
end

glx, glw = gausslaguerre(glm, reduced=true);
function H_SN(μ, σ, η)
    """
    Args:
        μ: center of skewed Gaussian
        σ: spread parameter
        η: skewness parameter

    Return:
        entropy of the skewed Gaussian
    """
    @assert length(μ) == length(σ) == length(η);
    k = length(μ);

    ϕ(x) = exp(-0.5*x^2.0) / sqrt(2π);
    Φ(x) = 0.5 * (1.0 + erf(x/√2));

    τ = norm(η);

    fp(x) = (f = 2*ϕ(x)*Φ( τ*x); (f != 0.0) && (f *= log(2*Φ( τ*x))); f);
    fm(x) = (f = 2*ϕ(x)*Φ(-τ*x); (f != 0.0) && (f *= log(2*Φ(-τ*x))); f);

    H0 = 0.5 * k + 0.5 * k * log(2π) + sum(log.(σ));

    return H0 - (sum(glw .* fp.(glx) .* exp.(glx)) + sum(glw .* fm.(glx) .* exp.(glx)));
end

# Qz, sample_Qz, H = Qzμσ0, sample_μσ, H_N;
# Qz, sample_Qz, H = Qzμσ1, sample_μσ, H_N;
Qz, sample_Qz, H = Qzμσ2, sample_μσ, H_N;
# Qz, sample_Qz, H = Qzμση0, sample_μση, H_SN;

function Equadform(X, Y)
    batch_size = size(Y,3);

    if length(X) == 0
        YZS = Y;
    else
        pZ = Qz(X, Y);

        if length(pZ) == 2
            μZ, σZ = pZ;
            EZS = cat(μZ..., dims=1);
        elseif length(pZ) == 3
            μZ, σZ, ηZ = pZ;
            ρZ = [rho(σZ_, ηZ_) for (σZ_,ηZ_) in zip(σZ,ηZ)];
            EZS = cat(μZ..., dims=1) + sqrt(2/π) * cat(ρZ..., dims=1);
        else
            error("unexpected length of pZ");
        end

        YZS = cat(Y,EZS, dims=1);
    end

    yzs = [vec(YZS[:,:,i]) for i in 1:batch_size];

    return mean(quadformSC(getα(), getβ(), yzs_; A=A, L=V) for yzs_ in yzs);
end

function Etrace(X, Y)
    batch_size = size(Y,3);

    pZ = Qz(X, Y);

    Γ = getΓ(getα(), getβ(); A=A);

    if length(pZ) == 2
        μZ, σZ = pZ;

        σZS = cat(σZ..., dims=1);
        diagΓUU = getdiagΓ(getα(), getβ(); A=A)[U];
        trace = j -> dot(diagΓUU, vec(σZS[:,:,j].^2.0));
    elseif length(pZ) == 3
        μZ, σZ, ηZ = pZ;
        ρZ = [rho(σZ_,ηZ_) for (σZ_,ηZ_) in zip(σZ,ηZ)];

        Var = (i,j,k) -> diagm(0=>σZ[k][:,i,j] .* σZ[k][:,i,j]) - 2/π * ρZ[k][:,i,j] * ρZ[k][:,i,j]';
        Gm(i,k) = (idx = FIDX(p1+ss[k]:p1+ff[k],[i]); Γ[idx,idx]);

        trace = j -> mapreduce(t->sum(Gm(t...) .* Var(t[1],j,t[2])), +, (i,k) for k in 1:p2 for i in 1:n; init=0.0);
    else
        error("unexpected length of pZ");
    end

    return mean(trace(j) for j in 1:batch_size);
end

function EH(X, Y)
    batch_size = size(Y,3);

    pZ = Qz(X, Y);

    if length(pZ) == 2
        μZ, σZ = pZ;

        HH = (i,j,k) -> H(μZ[k][:,i,j], σZ[k][:,i,j]);
    elseif length(pZ) == 3
        μZ, σZ, ηZ = pZ;

        HH = (i,j,k) -> H(μZ[k][:,i,j], σZ[k][:,i,j], ηZ[k][:,i,j]);
    else
        error("unexpected length of pZ");
    end

    return mapreduce(t->HH(t...), +, (i,j,k) for k in 1:p2 for j in 1:batch_size for i in 1:n; init=0.0) / batch_size;
end

function EQzlogPX(X, Y)
    batch_size = size(Y,3);
    ZS = sample_Qz(Qz(X,Y)...);

    return reduce(+, sum(logPX_ .* X_) for (logPX_,X_) in zip(logPX(ZS),X); init=0.0) / batch_size;
end

function loss(X, Y)
    Ω = 0.5 * logdetΓ(getα(), getβ(); A=A, P=V, t=t, k=k);
    Ω -= 0.5 * Equadform(X,Y);
    Ω -= 0.5 * Etrace(X,Y);
    Ω += EH(X,Y);
    Ω += EQzlogPX(X,Y);

    return -Ω/n;
end

dat = [(L->([X_[:,:,L] for X_ in X], Y[:,:,L]))(sample(1:N, n_batch)) for _ in 1:1000];

print_params() = @printf("α:  %s,  β:  %10.3f\n", array2str(getα()), getβ());
# ct = 0; print_params() = (global ct += 1; @printf("%5d,  loss:  %10.3f,  α:  %s,  β:  %10.3f,  μ:  %s,  logσ:  %s,  η:  %s\n", ct, loss(dat[end][1],dat[end][2]), array2str(getα()), getβ(), array2str(μ[1][:]), array2str(logσ[1][:]), array2str(η[1][:])));
train!(loss, Flux.params(φ, μ..., logσ..., η..., enc, reg), dat, ADAM(0.01); cb = print_params);

# function plot_SN!(h, μ, η, logσ; kwargs...)
#     ϕ(x) = exp(-0.5*x^2.0) / sqrt(2π);
#     Φ(x) = 0.5 * (1.0 + erf(x/√2));
#     f(x) = 2 * ϕ((x-μ)/exp(logσ)) * Φ(η*(x-μ)/exp(logσ));
#
#     Plots.plot!(h, -3.0:0.01:3.0, data.(f.(-3.0:0.01:3.0)) * 500; kwargs...);
# end
#
# h = Plots.plot(framestyle=:box);
# XV = reshape(X[1], (2, 8192));
# ZV = reshape(Z[1], (8192));
# histogram!(h, ZV[XV[1,:] .== 1], label="-, data", color=1);
# histogram!(h, ZV[XV[2,:] .== 1], label="+, data", color=2);
# plot_SN!(h, μ[1][1], η[1][1], logσ[1][1]; linewidth=5.0, label="-, learned", color="black");
# plot_SN!(h, μ[1][2], η[1][2], logσ[1][2]; linewidth=5.0, label="+, learned", color="grey");
# display(h);


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
