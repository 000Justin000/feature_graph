using Flux;
using Flux.Optimise;
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

tsctc(A, B) = reshape(A * reshape(B, (size(B,1), :)), (size(A,1), size(B)[2:end]...));

function get_ssff(d)
    ll = vcat(0, cumsum(d));
    ss = [ll[i]+1 for i in 1:length(d)];
    ff = [ll[i+1] for i in 1:length(d)];

    return ss, ff;
end

#-----------------------------------------------------------------------
# model dependent 1: compute the adjacency matrix of the graphical model
#-----------------------------------------------------------------------
function get_adjacency_matrices(G, p; interaction_list=vcat([(i,i) for i in 1:p], [(i,j) for i in 1:p for j in i+1:p]))
    """
    Given a graph, generate the graphical model that has every vertex mapped to
    p vertices, with p of them representing features

    Args:
       G: LightGraph Object
       p: number of features
       interaction_list: feature index pairs where there exist direct same-vertex interactions
    Return:
       A: an array of matrices for the graphical model:
          first p matrices: connections between same-channel features on
                            different vertices (normalized Laplacian)
          rest p(p+1) matrices: covariance among features on same vertices
    """
    n, L = nv(G), normalized_laplacian(G);

    A = Vector{SparseMatrixCSC}();

    # connections among corresponding features on different vertices
    # A_{i} = L ⊗ J_{ii}
    for i in 1:p
        push!(A, kron(L, sparse([i], [i], [1.0], p, p)));
    end

    # connections among different features on same vertices
    # A_{ii} = I ⊗ J_{ii}
    # A_{ij} = I ⊗ J_{ij}
    for (i,j) in interaction_list
        if (j == i)
            push!(A, kron(speye(n), sparse([i], [i], [1.0], p, p)));
        elseif (j>i)
            push!(A, kron(speye(n), sparse([i,j], [j,i], [1.0,1.0], p, p)));
        else
            error("unexpected pair")
        end
    end

    return A;
end

function getα(φ, p; interaction_list=vcat([(i,i) for i in 1:p], [(i,j) for i in 1:p for j in i+1:p]))
    """
    Log-Cholesky parametrization of the precision matrix

    Args:
       φ: p + q dimensional vector, q = p(p+1)/2 would indicate all pairwise interaction
       p: number of features
    """
    function upper_triangular(coeffs)
        Is = Vector{Int}();
        Js = Vector{Int}();
        Vs = Vector{eltype(φ)}();

        for (coeff,pair) in zip(coeffs,interaction_list)
            @assert pair[1] <= pair[2] "unexpected pair"
            push!(Is, pair[1]);
            push!(Js, pair[2]);
            push!(Vs, (pair[1] == pair[2]) ? exp(coeff) : coeff);
        end

        return Tracker.collect(sparse(Is,Js,Vs, p,p));
    end

    @assert (length(φ) == p + length(interaction_list)) "number of parameters mismatch number of matrices"

    R = upper_triangular(φ[p+1:end]);
    Q = R' * R;

    return vcat(exp.(φ[1:p]), Tracker.collect([Q[i,j] for (i,j) in interaction_list]));
end
#---------------------------------------------------------------------



#---------------------------------------------------------------------
# parameter used as decoder for synthetic data / initialize training
#---------------------------------------------------------------------
function init_W_(s, d; scale=1.0)
    if (s == 2 && d == 1)
        return reshape([-1.0, 1.0] * scale, (2,1));
    elseif (s == d)
        return diagm(0 => ones(s)) * scale;
    else
        error("unexpected (s,d) pair");
    end
end
init_b_(s) = zeros(s);
#---------------------------------------------------------------------

function prepare_data(dataset; N=1, p1=1, p2=1, s=[2], d=[1])
    @assert (p2 == length(s) == length(d)) "number of categorical variables mismatch"

    if ((options = match(r"synthetic_([a-z]+)", dataset)) != nothing)
        if options[1] == "tiny"
            G = complete_graph(2);
        elseif options[1] == "small"
            G = watts_strogatz(10, 6, 0.30);
        elseif options[1] == "ring"
            G = watts_strogatz(300, 6, 0.00);
        elseif options[1] == "medium"
            G = watts_strogatz(500, 6, 0.03);
        elseif options[1] == "large"
            G = watts_strogatz(3000, 6, 0.01);
        else
            error("unexpected size option");
        end

        n = nv(G);
        p = p1 + sum(d);
        interaction_list=vcat([(i,i) for i in 1:p], [(i,j) for i in 1:p for j in i+1:p]);
        A = get_adjacency_matrices(G, p; interaction_list=interaction_list);

        T = randn(p+1,p); Q = inv((T'*T)/(p+1));
        α0 = vcat(exp.(randn(p)), [Q[i,j] for (i,j) in interaction_list]);
        @printf("α0: %s\n", array2str(α0)); flush(stdout);

        cr = [p1+ss_:p1+ff_ for (ss_,ff_) in zip(get_ssff(d)...)];

        CM0 = inv(Array(getΓ(α0; A=A)));
        CM = (CM0 + CM0')/2.0;
        g = MvNormal(CM);
        YZ = cat([reshape(rand(g), (p,n)) for _ in 1:N]..., dims=3);
        Y = YZ[1:p1,:,:];
        Z = [YZ[cr_,:,:] for cr_ in cr];

        W0 = [param(init_W_(s[i], d[i])) for i in 1:p2];
        b0 = [param(init_b_(s[i])) for i in 1:p2];

        logPX(Z) = [logsoftmax(tsctc(W_,Z_) .+ repeat(b_,1,n,size(Z_,3)), dims=1) for (W_,b_,Z_) in zip(W0,b0,Z)];

        X = [begin
                x = zeros(size(logPX_));
                for j in 1:size(logPX_,2)
                    for k in 1:size(logPX_,3)
                        x[argmax(logPX_[:,j,k]),j,k] = 1;
                    end
                end
                x;
             end for logPX_ in logPX(Z)];
    elseif ((match(r"county_(.+)_([0-9]+)", dataset) != nothing) ||
            (match(r"environment_(.+)_([0-9]+)", dataset) != nothing) ||
            (match(r"ward_(.+)_([0-9]+)", dataset) != nothing))
        G, _, labels, feats = read_network(dataset);
        for i in vertices(G)
            rem_edge!(G, i,i);
        end

        n = nv(G);
        p1, p2 = length(feats[1]) + length(labels[1]), 0;
        s, d = Int[], Int[];
        p = p1 + sum(d);
        interaction_list=vcat([(i,i) for i in 1:p], [(i,j) for i in 1:p for j in i+1:p])
        A = get_adjacency_matrices(G, p; interaction_list=interaction_list);

        α0 = nothing;
        Z = nothing;
        Y = unsqueeze(hcat([vcat(feat,label) for (feat,label) in zip(feats,labels)]...), 3);
        X = [];
    elseif (match(r"twitch_(.+)_true_([0-9]+)", dataset) != nothing)
        G, _, labels, feats = read_network(dataset);
        for i in vertices(G)
            rem_edge!(G, i,i);
        end

        n = nv(G);
        p1, p2 = length(feats[1]) + length(labels[1]), 0;
        s, d = Int[], Int[];
        p = p1 + sum(d);
        interaction_list=vcat([(i,i) for i in 1:p], [(i,j) for i in 1:p for j in i+1:p])
        A = get_adjacency_matrices(G, p; interaction_list=interaction_list);

        α0 = nothing;
        Z = nothing;
        Y = unsqueeze(hcat([vcat(feat,label) for (feat,label) in zip(feats,labels)]...), 3);
        X = [];
    elseif (match(r"cora_true_([0-9]+)", dataset) != nothing)
        G, _, labels, feats = read_network(dataset);
        for i in vertices(G)
            rem_edge!(G, i,i);
        end

        n = nv(G);
        p1, p2 = length(feats[1]), 1;
        s, d = Int[7], Int[7];
        p = p1 + sum(d);
        interaction_list=vcat([(i,i) for i in 1:p], [(i,j) for i in 1:p for j in i+1:p])
        A = get_adjacency_matrices(G, p; interaction_list=interaction_list);

        α0 = nothing;
        Z = nothing;
        Y = unsqueeze(hcat([feat for feat in feats]...), 3);
        X = [unsqueeze(hcat([eye(7)[:,label] for label in labels]...), 3)];
    elseif (match(r"cora_false_([0-9]+)", dataset) != nothing)
        G, _, labels, feats = read_network(dataset);
        for i in vertices(G)
            rem_edge!(G, i,i);
        end

        n = nv(G);
        p1, p2 = 1433, 1;
        s, d = [7], [7];
        p = p1 + sum(d);
        interaction_list = [];
        A = [];
        α0 = nothing;
        X = [unsqueeze(hcat([eye(7)[:,label] for label in labels]...), 3)];
        Y = f32(hcat(feats...) .- mean(feats));
        Z = nothing;
    elseif (match(r"cropsim_([a-z]+)_([0-9]+)_([0-9]+)", dataset) != nothing)
        G, _, labels, feats = read_network(dataset);
        for i in vertices(G)
            rem_edge!(G, i,i);
        end

        n = nv(G);
        p1, p2 = length(feats[1]) + length(labels[1]), 0;
        s, d = Int[], Int[];
        p = p1 + sum(d);
        interaction_list=[];
        A = [];
        α0 = nothing;
        X = [];
        Y = unsqueeze(hcat([vcat(feat,label) for (feat,label) in zip(feats,labels)]...), 3);
        Z = nothing;
    elseif (match(r"(YelpChi|Amazon)_([0-9]+)", dataset) != nothing)
        G, _, labels, feats = read_network(dataset);
        for i in vertices(G)
            rem_edge!(G, i,i);
        end

        n = nv(G);
        p1, p2 = length(feats[1]), 1;
        s, d = Int[2], Int[2];
        p = p1 + sum(d);
        interaction_list=vcat([(i,i) for i in 1:p], [(i,j) for i in 1:p for j in i+1:p])
        A = get_adjacency_matrices(G, p; interaction_list=interaction_list);

        α0 = nothing;
        Z = nothing;
        Y = f32(hcat(feats...) .- mean(feats));
        X = [unsqueeze(hcat([eye(2)[:,label] for label in labels]...), 3)];
    end

    λ_getα = φ -> getα(φ, p; interaction_list=interaction_list);

    return G, A, λ_getα, s, d, α0, X, Y, Z;
end

# ARGS = ["0", "county_election_2012",       "true",     "7:7",    "0.1:0.1:0.6", "0.9"];
# ARGS = ["0", "ward_election_2012",         "true",     "6:6",    "0.1:0.1:0.6", "0.9"];
# ARGS = ["0", "environment_pm2.5_2008",     "true",     "5:5",    "0.1:0.1:0.6", "0.9"];
# ARGS = ["0", "twitch_PTBR_true_4",         "true",     "5:5",    "0.1:0.1:0.6", "0.9"];
# ARGS = ["0", "cora_true_8",                "true",     "9:15",   "0.1:0.1:0.6", "0.9"];
# ARGS = ["0", "cora_false_0",               "false", "1434:1440", "0.1:0.1:0.6", "0.9"];
# ARGS = ["0", "cropsim_harvestarea_2000_5", "false",    "6:6",    "0.1:0.1:0.6", "0.9"];

seed_val = Meta.parse(ARGS[1]) |> eval;
Random.seed!(seed_val);

dataset = ARGS[2];
encoder = ["MAP", "GNN"][2];
Qform = ["N", "SN"][1];
t, k, glm, dim_h = 128, 32, 100, 32;
N = 1;
n_batch = 1;

# note G is the entity graph, A is the adjacency matrices for the graphical model
G, A, λ_getα, s, d, α0, X, Y, Z = prepare_data(dataset; N=N, p1=0, p2=3, s=Int[2,2,2], d=Int[1,1,1]);

n = nv(G);
p1 = size(Y,1);
p2 = length(X);
p = p1 + sum(d);

# the indices for features in fidx and vertices in V
FIDX(fidx, V=vertices(G)) = [(i-1)*p+j for i in V for j in fidx];



function fit_model(seed_val=0)
    Random.seed!(seed_val);

    n_step = 1000;

    V = collect(1:size(A[1],1));
    L = FIDX(1:p1);
    U = setdiff(V,L);

    # parameter for the decoder (W fixed, b can change)
    W = [param(init_W_(s[i], d[i])) for i in 1:p2];
    b = [param(init_b_(s[i])) for i in 1:p2];

    # parameter for the encoder (to be learned)
    μ = [param(zeros(d[j], s[j])) for j in 1:p2];
    logσ = [param(zeros(d[j], s[j])) for j in 1:p2];
    η = [param(zeros(d[j], s[j])) for j in 1:p2];
    enc = graph_encoder(p1+sum(s), dim_h, dim_h, repeat(["SAGE_Mean"], 1); σ=relu);
    reg = Dense(dim_h, sum(d) * Dict("N" => 2, "SN" => 3)[Qform]);

    logPX(Z) = [logsoftmax(tsctc(W_,Z_) .+ repeat(b_,1,n,size(Z_,3)), dims=1) for (W_,b_,Z_) in zip(W,b,Z)];

    #---------------------------------------------------------------------
    # model dependent 2: from flux parameters to α
    #---------------------------------------------------------------------
    φ = param(zeros(length(A)));
    getα() = λ_getα(φ);
    #---------------------------------------------------------------------

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

        return mean(quadformSC(getα(), yzs_; A=A, L=V) for yzs_ in yzs);
    end

    function Equadform(Z)
        ZS = cat(Z..., dims=1);
        batch_size = size(ZS,3);
        yzs = [vec(ZS[:,:,i]) for i in 1:batch_size];

        return mean(quadformSC(getα(), yzs_; A=A, L=V) for yzs_ in yzs);
    end

    function Etrace(X, Y)
        batch_size = size(Y,3);

        pZ = Qz(X, Y);

        Γ = getΓ(getα(); A=A);

        if length(pZ) == 2
            μZ, σZ = pZ;

            σZS = cat(σZ..., dims=1);
            diagΓUU = getdiagΓ(getα(); A=A)[U];
            trace = j -> (length(U) == 0) ? 0 : dot(diagΓUU, vec(σZS[:,:,j].^2.0));
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
        Ω = 0.5 * logdetΓ(getα(); A=A, P=V, t=t, k=k);
        Ω -= 0.5 * Equadform(X,Y);
        Ω -= 0.5 * Etrace(X,Y);
        Ω += EH(X,Y);
        Ω += EQzlogPX(X,Y);

        return -Ω/n;
    end

    dat = [(L->([X_[:,:,L] for X_ in X], Y[:,:,L]))(sample(1:N, n_batch)) for _ in 1:n_step];
    print_params() = (@printf("α:  %s\n", array2str(getα())); flush(stdout));
    train!(loss, [Flux.params(φ, μ..., logσ..., η...), Flux.params(enc, reg)], dat, [ADAM(1.0e-2), ADAM(1.0e-2)]; start_opts = [0, 0], cb = print_params, cb_skip=100);

    return data(getα());
end

fitα = Meta.parse(ARGS[3]) |> eval;
α = fitα ? fit_model(seed_val) : nothing;

lidx = Meta.parse(ARGS[4]) |> eval;
fidx = setdiff(collect(1:p), lidx);

# analysis
function print_MI(lidx, fidx, ll, uu; seed_val=0)
    Random.seed!(seed_val);

    obsl = FIDX(lidx, ll); obsf = FIDX(fidx, vertices(G)); trgl = FIDX(lidx, uu);
    getH_L2()     = (-logdetΓ(param(α); A=A, P=vcat(obsf,obsl,trgl), t=t, k=k)) - (-logdetΓ(param(α); A=A, P=vcat(obsf,obsl), t=t, k=k));
    getH_L2_L1()  = (-logdetΓ(param(α); A=A, P=vcat(obsf,trgl), t=t, k=k)) - (-logdetΓ(param(α); A=A, P=obsf, t=t, k=k));
    getH_L2_F()   = (-logdetΓ(param(α); A=A, P=vcat(obsl,trgl), t=t, k=k)) - (-logdetΓ(param(α); A=A, P=obsl, t=t, k=k));
    getH_L2_FL1() = (-logdetΓ(param(α); A=A, P=trgl, t=t, k=k));

    H_L2     = mean([data(getH_L2())     for _ in 1:30]); # entropy of target labels, marginalized over observed features, observed labels
    H_L2_L1  = mean([data(getH_L2_L1())  for _ in 1:30]); # entropy of target lables, conditioned on observed labels, marginalized over observed features
    H_L2_F   = mean([data(getH_L2_F())   for _ in 1:30]); # entropy of target lables, conditioned on observed features, marginalized over observed labels
    H_L2_FL1 = mean([data(getH_L2_FL1()) for _ in 1:30]); # entropy of target lables, conditioned on observed features, observed labels

    @printf("    LP MI:    %10.4f\n", (H_L2 - H_L2_L1)  / length(uu)); flush(stdout);
    @printf("   SLR MI:    %10.4f\n", (H_L2 - H_L2_F)   / length(uu)); flush(stdout);
    @printf("SLR_LP MI:    %10.4f\n", (H_L2 - H_L2_FL1) / length(uu)); flush(stdout);
end

# learn and test accuracy
function run_dataset(G, feats, labels, ll, uu; predictor="zero", correlation="zero", feature_smoothing=true, seed_val=0, prefix="")

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
        rU = vcat([cg(Γ[U,U], -Γ[U,L]*rL[i,:])' for i in 1:size(rL,1)]...);

        r = rL * expansion(n,L)' + rU * expansion(n,U)';

        return r;
    end

    function predict(U, L; labelL, getPrediction, Γ)
        """
        Args:
                    U: vertices to predict
                    L: vertices with ground truth labels
               labelL: ground truth labels on L
        getPrediction: base predictor function
                    Γ: label propagation matrix

        Returns:
                   lU: predictive label = base predictor output + estimated noise
        """

        pUL = getPrediction(vcat(U, L));
        pU = pUL[:,1:length(U)];
        pL = pUL[:,length(U)+1:end];

        # for regression task, residual is defined as ``true-label minus predicted-label''
        if (size(labelL,1) == 1)
            rL = (labelL - data(pL));
        # for classification task, if the prediction is correct, residual should be 0
        else
            gap = 0.8;

            function process(x::Vector, loc::Int)
                @assert (all(x .>= 0) && abs(sum(x) - 1.0) < 1.0e-6) "unexpected probability vector"

                c = sortperm(x; rev=true);
                if ((c[1] == loc) && (x[c[1]] - x[c[2]] >= gap))
                    return x;
                else
                    xx = x[:];
                    c0 = setdiff(c, loc);

                    xx[c0] *= (1-gap) / (x[c0[1]] + sum(x[c0]));
                    xx[loc] = xx[c0[1]] + gap;

                    return xx;
                end
            end

            hpL = collect(pL);
            for i in 1:length(L)
                hpL[:,i] = process(hpL[:,i], argmax(labelL[:,i]));
            end

            rL = hpL .- collect(pL);
        end

        lU = data(pU) + interpolate(L, rL; Γ=Γ)[:,U];

        return lU;
    end

    function smooth(feats, S; η=0.9, K=300)
        results = zeros(size(feats));
        smoothed_feats = feats;
        for i in 0:K-1
            results += (1-η)*η^i * smoothed_feats;
            smoothed_feats *= S;
        end
        results += η^K * smoothed_feats;

        return results;
    end

    @assert !(predictor == "gnn" && feature_smoothing) "feature smoothing & gnn are redundant"

    Random.seed!(seed_val);
    n_batch = Int(ceil(length(ll)*0.1));
    classification = (size(labels,1) != 1);
    accuracyFun = classification ? detection_f1 : R2;

    if length(ARGS) >= 6
        η = parse(Float64, ARGS[6]);
    elseif α != nothing
        η = mean(α[lidx]./(α[lidx]+α[p.+lidx]));
    else
        η = 0.9;
    end

    S = spdiagm(0=>degree(G).^-0.5) * adjacency_matrix(G) * spdiagm(0=>degree(G).^-0.5);
    feature_smoothing && (feats = smooth(feats, S; η = η));

    if predictor == "zero"
        getPrediction = classification ? L -> ones(size(labels,1),length(L)) / size(labels,1) : L -> zeros(size(labels,1),length(L));
        θ = Flux.params();
        optθ = ADAM(0.0);
    elseif predictor == "linear"
        lls = Chain(Dense(size(feats,1), size(labels,1)), classification ? softmax : identity);
        getPrediction = L -> lls(feats[:,L]);
        θ = Flux.params(lls);
        optθ = Optimiser(WeightDecay(1.0e-3), ADAM(1.0e-2));
    elseif predictor == "mlp"
        mlp = Chain(Dense(size(feats,1), dim_h, relu), Dense(dim_h, dim_h, relu), Dense(dim_h, dim_h, relu), Dense(dim_h, size(labels,1)), classification ? softmax : identity);
        getPrediction = L -> mlp(feats[:,L]);
        θ = Flux.params(mlp);
        optθ = Optimiser(WeightDecay(1.0e-4), ADAM(1.0e-3));
    elseif predictor == "gnn"
        enc = graph_encoder(size(feats,1), dim_h, dim_h, repeat(["SAGE_Mean"], 2); ks=repeat([5], 2), σ=relu);
        reg = Chain(Dense(dim_h, size(labels,1)), classification ? softmax : identity);
        getPrediction = L -> reg(hcat(enc(G, L, u->feats[:,u])...));
        θ = Flux.params(enc, reg);
        optθ = Optimiser(WeightDecay(1.0e-4), ADAM(1.0e-3));
    else
        error("unexpected predictor type");
    end

    if correlation == "zero"
        Γ = speye(n);
    elseif correlation == "homo"
        # Γ = speye(n) + (η / (1 - η))*normalized_laplacian(G);
        Γ = normalized_laplacian(G);
    else
        error("unexpected correlation");
    end

    function loss(L)
        if classification
            weights = sum(labels .* (1 ./ sum(labels, dims=2)), dims=1)[:];
            normalized_weights = ones(size(labels,1)) * (weights ./ mean(weights))';

            return Flux.crossentropy(getPrediction(L), labels[:,L], weight=normalized_weights[:,L]);
        else
            return Flux.mse(getPrediction(L), labels[:,L]);
        end
    end

    n_step = 300; mini_batches = [tuple(sample(ll, n_batch, replace=false)) for _ in 1:n_step];

    cb() = (@printf("%6.3f,    %6.3f,    %6.3f\n", loss(ll), loss(uu), accuracyFun(predict(uu,ll; labelL=labels[:,ll], getPrediction=getPrediction, Γ=Γ), labels[:,uu])); flush(stdout));
    (predictor != "zero") && train!(loss, [θ], mini_batches, [optθ]; cb=()->nothing, cb_skip=100);

    @printf("%s AC:    %10.4f\n", prefix, accuracyFun(predict(uu,ll; labelL=labels[:,ll], getPrediction=getPrediction, Γ=Γ), labels[:,uu]));
    flush(stdout);
end

split_ratios = Meta.parse(ARGS[5]) |> eval;
for split_ratio in split_ratios
    for seed_increment in 1:1
        Random.seed!(seed_val + seed_increment);

        ll, uu = rand_split(nv(G), split_ratio);
        fitα && print_MI(lidx, fidx, ll, uu; seed_val=seed_val);

        feats, labels = Y[fidx,:,1], ((length(X) == 0) ? Y[lidx,:,1] : X[1][:,:,1]);
        run_dataset(G, feats, labels, ll, uu, predictor="zero",   correlation="homo", feature_smoothing=false, seed_val=seed_val, prefix="    LP");
        run_dataset(G, feats, labels, ll, uu, predictor="linear", correlation="zero", feature_smoothing=false, seed_val=seed_val, prefix="    LR");
        run_dataset(G, feats, labels, ll, uu, predictor="linear", correlation="zero", feature_smoothing=true,  seed_val=seed_val, prefix="   SLR");
        run_dataset(G, feats, labels, ll, uu, predictor="linear", correlation="homo", feature_smoothing=false, seed_val=seed_val, prefix=" LR_LP");
        run_dataset(G, feats, labels, ll, uu, predictor="linear", correlation="homo", feature_smoothing=true,  seed_val=seed_val, prefix="SLR_LP");
        run_dataset(G, feats, labels, ll, uu, predictor="gnn",    correlation="zero", feature_smoothing=false, seed_val=seed_val, prefix="   GNN");
        run_dataset(G, feats, labels, ll, uu, predictor="gnn",    correlation="homo", feature_smoothing=false, seed_val=seed_val, prefix="GNN_LP");
    end
    @printf("\n"); flush(stdout);
end
