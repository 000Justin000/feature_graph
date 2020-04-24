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
using MLMetrics;

include("utils.jl");
include("kernels.jl");
include("read_network.jl");
include("common.jl");

function interpolate(L, rL; Γ)
    """
    Args:
         L: mini_batch indices for estimating noise
        rL: noise over the mini_batch L
         Γ: label propagation matrix

    Returns:
         r: noise over all vertices
    """
    n = size(Γ,1);
    U = setdiff(1:n, L);
    rU = hcat([cg(Γ[U,U], -Γ[U,L]*rL[:,i]) for i in 1:size(rL,2)]...);

    r = expansion(n,L) * rL + expansion(n,U) * rU;

    return r;
end

function pred(U, L; labelL, predict, Γ)
    """
    Args:
          U: vertices to predict
          L: vertices with ground truth labels
     labelL: ground truth labels on L
    predict: base predictor function
          Γ: label propagation matrix

    Returns:
         lU: predictive label = base predictor output + estimated noise
    """

    pUL = predict(vcat(U, L));
    pU = pUL[:,1:length(U)];
    pL = pUL[:,length(U)+1:end];

    rL = labelL - data(pL);
    lU = pU + interpolate(L, rL'; Γ=Γ)[U,:]';

    return lU;
end

# Random.seed!(parse(Int,ARGS[1]));
Random.seed!(0);

dataset = "cora_false_0";
dim_h, dim_r = 16, 8;
n_step = 200;
ptr = 0.01;
model = ["uniform", "gnn"][1];
correlation = ["zero", "homo"][2];

if ((options = match(r"synthetic_([0-9]+)", dataset)) != nothing)
    # note G is the entity graph, A is the adjacency matrices for the graphical model
    G, _, Y, X, _, _ = prepare_data(dataset; N=1, p1=0, p2=3, s=[2,2,2], d=[1,1,1]);
    feats = vcat(X[1], X[2])[:,:,1];
    labels = X[3][:,:,1];
    n = nv(G);
elseif ((options = match(r"cora_true_([0-9]+)", dataset)) != nothing)
    G, _, y, f = read_network(dataset);
    feats = hcat(f...);
    labels = Flux.onehotbatch(y, 1:7);
    n = nv(G);
elseif ((options = match(r"cora_false_([0-9]+)", dataset)) != nothing)
    G, _, y, f = read_network(dataset);
    feats = hcat(f...);
    labels = Flux.onehotbatch(y, 1:7);
    n = nv(G);
else
    error("unexpected dataset");
end

if model == "uniform"
    base_prob(L) = ones(size(labels,1), length(L)) / size(labels,1);
elseif model == "gnn"
    enc = graph_encoder(size(feats,1), dim_r, dim_h, repeat(["SAGE_Mean"], 2); σ=relu);
    reg = Chain(Dense(dim_r, size(labels,1)), softmax);
    base_prob(L) = reg(hcat(enc(G, L, u->feats[:,u])...));
else
    error("unexpected model");
end

if correlation == "zero"
    Γ = speye(n);
elseif correlation == "homo"
    Γ = float(laplacian_matrix(G));
else
    error("unexpected correlation");
end

loss(L) = Flux.crossentropy(base_prob(L), labels[:,L]);
acrc(U,L) = (probs = pred(U,L; labelL=labels[:,L], predict=base_prob, Γ=Γ); sum(labels[:,U][argmax(data(probs), dims=1)]) / length(U));
f1sc(U,L) = (probs = pred(U,L; labelL=labels[:,L], predict=base_prob, Γ=Γ); f_score(map(ci->ci.I[1], argmax(labels[:,U], dims=1)[:]), map(ci->ci.I[1], argmax(data(probs), dims=1)[:]); avgmode=:micro));

L, VU = rand_split(n, ptr);
V, U = VU[1:div(length(VU),2)], VU[div(length(VU),2)+1:end];

n_batch = Int(round(length(L) * 0.50));
mini_batches = [tuple(sample(L, n_batch, replace=false)) for _ in 1:n_step];

cb() = @printf("%6.3f,    %6.3f,    %6.3f,    %6.3f\n", loss(L), loss(V), acrc(V,L), f1sc(V,L));
train!(loss, [Flux.params(enc, reg)], mini_batches, [ADAM(0.001)]; cb=cb, cb_skip=10);

@printf("%6.3f\n", acrc(U,L));
