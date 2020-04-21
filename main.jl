using Flux;
using Flux: train!, throttle, Tracker, unsqueeze;
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
include("read_network.jl");
include("common.jl");

Random.seed!(parse(Int,ARGS[1]));

dataset = "synthetic_medium";
encoder = ["MAP", "GNN", "HEU"][2];
Qform = ["N", "SN"][1];
t, k, glm, dim_h, dim_r = 128, 32, 100, 32, 8;
N = 1;
n_batch = 1;
n_step = 10000;

# note G is the entity graph, A is the adjacency matrices for the graphical model
G, A, Y, X, s, d = prepare_data(dataset; N=N, p1=0, p2=3, s=[2,2,2], d=[1,1,1]);

n = nv(G);
p1 = size(Y,1);
p2 = length(X);
p = p1 + sum(d);

FIDX(fidx, V=vertices(G)) = [(i-1)*p+j for i in V for j in fidx];

V = collect(1:size(A[1],1));
L = FIDX(1:p1);
U = setdiff(V,L);

# parameter for the decoder (fixed)
W = [param(init_W_(s[i], d[i])) for i in 1:p2];
b = [param(init_b_(s[i])) for i in 1:p2];

# parameter for the encoder (to be learned)
μ = [param(zeros(d[j], s[j])) for j in 1:p2];
logσ = [param(zeros(d[j], s[j])) for j in 1:p2];
η = [param(zeros(d[j], s[j])) for j in 1:p2];
enc = graph_encoder(p1+sum(s), dim_r, dim_h, repeat(["SAGE_Mean"], 2); σ=relu);
reg = Dense(dim_r, sum(d) * Dict("N" => 2, "SN" => 3)[Qform]);

logPX(Z) = [logsoftmax(tsctc(W_,Z_) .+ repeat(b_,1,n,size(Z_,3)), dims=1) for (W_,b_,Z_) in zip(W,b,Z)];

# parameter for the latent Gaussian (to be learned)
φ = param(zeros(length(A)+1));
getα() = φ[1:end-1];
# getβ() = exp(φ[end]);
getβ() = param(1.0);

# return normal distribution with diagonal covariance matrix, conditioned on X
function Qzμσ0(X, Y)
    μZ0 = [tsctc(μ_, X_) for (μ_, X_) in zip(μ, X)];
    σZ0 = [exp.(tsctc(logσ_, X_)) for (logσ_, X_) in zip(logσ, X)];

    return μZ0, σZ0;
end

# use GNN to learn μZ1 and σZ1
function Qzμσ1(X, Y)
    batch_size = size(Y,3);

    YX = cat(Y, X...; dims=1);
    μlogσZ = Flux.stack([reg(hcat(enc(G, collect(1:n), u->YX[:,u,i])...)) for i in 1:batch_size], 3);

    μZ, σZ = μlogσZ[1:sum(d),:,:], exp.(μlogσZ[sum(d)+1:sum(d)*2,:,:]);
    μZ1 = [μZ[ss_:ff_,:,:] for (ss_,ff_) in zip(get_ssff(d)...)];
    σZ1 = [σZ[ss_:ff_,:,:] for (ss_,ff_) in zip(get_ssff(d)...)];

    return μZ1, σZ1;
end

# return normal distribution with diagonal covariance matrix, conditioned on X & Y
function Qzμσ2(X, Y)
    batch_size = size(Y,3);

    μZ0, σZ0 = Qzμσ0(X, Y);

    C1 = [σZ0_.^-2.0 .* μZ0_ for (μZ0_,σZ0_) in zip(μZ0,σZ0)];

    C2S = reshape(-ΓX(getα(), getβ(), reshape(Y, (p1*n, batch_size)); A=A, U=U, L=L), (sum(d), n, batch_size));
    C2 = [C2S[ss_:ff_,:,:] for (ss_,ff_) in zip(get_ssff(d)...)];

    σZ2 = [(σZ0_.^-2.0 .+ getβ()).^-0.5 for σZ0_ in σZ0];
    μZ2 = [σZ2_.^2.0 .* (C1_ .+ C2_) for (σZ2_, C1_, C2_) in zip(σZ2, C1, C2)];

    return μZ2, σZ2;
end

# sample normal distribution with diagonal covariance matrix
function sample_μσ(μZ, σZ)
    ZS = [μZ_ + σZ_ .* randn(size(σZ_)) for (μZ_, σZ_) in zip(μZ, σZ)];

    return ZS;
end

# intermediate variable for skewed Gaussian computation
rho(σZ_, ηZ_) = (σZ_ .* ηZ_) ./ sqrt.(1 .+ sum(ηZ_ .* ηZ_, dims=1));

# return skewed normal distribution with diagonal covariance matrix, conditioned on X
function Qzμση0(X, Y)
    μZ0 = [tsctc(μ_, X_) for (μ_, X_) in zip(μ, X)];
    σZ0 = [exp.(tsctc(logσ_, X_)) for (logσ_, X_) in zip(logσ, X)];
    ηZ0 = [tsctc(η_, X_) for (η_, X_) in zip(η, X)];

    return μZ0, σZ0, ηZ0;
end

# use GNN to learn μZ1, σZ1, and ηZ1
function Qzμση1(X, Y)
    batch_size = size(Y,3);

    YX = cat(Y, X...; dims=1);
    μlogσηZ = Flux.stack([reg(hcat(enc(G, collect(1:n), u->YX[:,u,i])...)) for i in 1:batch_size], 3);

    μZ, σZ, ηZ = μlogσηZ[1:sum(d),:,:], exp.(μlogσηZ[sum(d)+1:sum(d)*2,:,:]), μlogσηZ[sum(d)*2+1:sum(d)*3,:,:];
    μZ1 = [μZ[ss_:ff_,:,:] for (ss_,ff_) in zip(get_ssff(d)...)];
    σZ1 = [σZ[ss_:ff_,:,:] for (ss_,ff_) in zip(get_ssff(d)...)];
    ηZ1 = [ηZ[ss_:ff_,:,:] for (ss_,ff_) in zip(get_ssff(d)...)];

    return μZ1, σZ1, ηZ1;
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
                ZS_[:,j,k] .= μZ_[:,j,k] .+ ρZ_[:,j,k] * abs(randn(Float32)) + chol(CM_) * randn(Float32, d_);
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

if Qform == "N"
    H, sample_Qz = H_N, sample_μσ;

    if encoder == "MAP"
        Qz = Qzμσ0;
    elseif encoder == "GNN"
        Qz = Qzμσ1
    elseif encoder == "HEU"
        Qz = Qzμσ2;
    else
        error("unsupported encoder")
    end
elseif Qform == "SN"
    H, sample_Qz = H_SN, sample_μση;

    if encoder == "MAP"
        Qz = Qzμση0;
    elseif encoder == "GNN"
        Qz = Qzμση1
    else
        error("unsupported encoder")
    end
else
    error("unsupported Qform")
end

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
        ss, ff = get_ssff(d);
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

dat = [(L->([X_[:,:,L] for X_ in X], Y[:,:,L]))(sample(1:N, n_batch)) for _ in 1:n_step];

print_params() = @printf("α:  %s,    β:  %10.3f\n", array2str(getα()), getβ());
# ct = 0; print_params() = (global ct += 1; @printf("%5d,  loss:  %10.3f,  α:  %s,  β:  %10.3f,  μ:  %s,  logσ:  %s,  η:  %s\n", ct, loss(dat[end][1],dat[end][2]), array2str(getα()), getβ(), array2str(μ[1][:]), array2str(logσ[1][:]), array2str(η[1][:])));
# train!(loss, Flux.params(φ, μ..., logσ..., η..., enc, reg), dat, ADAM(0.01); cb = throttle(print_params,1));
train!(loss, [Flux.params(φ, μ..., logσ..., η...), Flux.params(enc, reg)], dat, [Descent(1.0e-2), ADAM(1.0e-2)]; start_opts = [Int(n_step/5), 0], cb = print_params, cb_skip=10);

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
