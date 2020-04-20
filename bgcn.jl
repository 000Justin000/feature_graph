using Random;
using Statistics;
using StatsBase: sample, randperm, mean;
using LinearAlgebra;
using SparseArrays;
using IterativeSolvers;
using LightGraphs;
using Flux;
using GraphSAGE;
using Printf;
using Plots;
using PyCall;

include("example_networks.jl");
include("utils.jl");
include("precision.jl");
include("extras.jl");

# network_name0, network_name1 = "ising_35_0.10_0.35", "ising_50_0.10_0.35";
# network_name0, network_name1 = "ising_35_0.10_-0.35", "ising_50_0.10_-0.35";
network_name0, network_name1 = "county_election_2012", "county_election_2016";
# network_name0, network_name1 = "county_unemployment_2012", "county_unemployment_2016";
# network_name0, network_name1 = "Anaheim", "ChicagoSketch";
# network_name0, network_name1 = "ChicagoSketch", "Anaheim";
# network_name0, network_name1 = "twitch_PTBR_true_32", "twitch_RU_true_32";
# network_name0, network_name1 = "wikipedia_chameleon_true_8", "wikipedia_chameleon_true_8";
# network_name0, network_name1 = "facebook100_Swarthmore42", "facebook100_Amherst41";
dim_out, dim_h = 8, 16;
t, k, num_steps = 30, 50, 1500;
update_αβ, inductive, label_feat = false, false, false;
ptr_inductive = 0.6;
ϵ, δ, η = 0.0e-6, 0.0e-6, 1.0e-6;
accuracyFun = R2;
num_ave = 10;

ptrs = 0.60:0.05:0.60;
rrt = Vector{Float64}();
rri = Vector{Float64}();

for (i, ptr) in enumerate(ptrs)
    for seed_val in 1:num_ave
        Random.seed!(seed_val);
        println("\n\nseed_val:    ", seed_val);

        G, A, labels, feats = read_network(network_name0); n = nv(G);
        d = sum(sum(A), dims=1)[:];

        # Label as Feature
        label_feat && (feats = [vcat(feat,label) for (feat,label) in zip(feats,labels)]);

        L, U = rand_split(n, ptr);
        enc = graph_encoder(length(feats[1]), dim_out, dim_h, repeat(["SAGE_Mean"], 2); σ=relu);
        reg = Dense(dim_out, 1);
        pcs = PCS(zeros(length(A)), 0.0, η, spdiagm(0=>d), A);

        # Label as Feature
        function masked_feats(feats, UU, L, label_feat)
            UUSet = Set(UU);
            LSet = Set(L);
            if !label_feat
                return feats;
            else
                mean_label = mean([feat[end] for (i,feat) in enumerate(feats) if in(i,LSet) && !in(i,UUSet)]);
                return [in(i,LSet) && !in(i,UUSet) ? feat : vcat(feat[1:end-1], mean_label) for (i,feat) in enumerate(feats)];
            end
        end

        mutable struct call_back
            i::Int;
        end

        function (c::call_back)()
            c.i += 1;

            if c.i == 1
                @printf("\naccuracy,    α,    β\n");
            end

            if c.i % 100 == 0
                if update_αβ
                    p = vcat(reg.(enc(G, collect(1:n), u->feats[u]))...);
                    rL = labels[L] - p[L].data;
                    dL = sum(sum(A)[L,L], dims=1)[:];

                    # the number of edges between vertices in the training set
                    rAr = [sum(A[i][L,L]) == 0 ? 0 : dot(rL, A[i][L,L]*rL)/sum(A[i][L,L]) * sum(A[i]) for i in 1:length(A)];
                    rDr = sum(dL) == 0 ? 0 : sum(rL.^2 .* dL)/sum(dL) * sum(sum(A));
                    @assert rDr >= abs(sum(rAr));
                    rIr = sum(rL.^2)/length(L) * n;

                    update_αβ!(pcs, rAr, rDr, rIr; t=t, k=k, ϵ=ϵ, δ=δ)
                end

                @printf("%5.2f,    %s,    %5.2f\n", accuracyFun(labels[U], pred(U, L; G=G,feats=masked_feats(feats,U,L,label_feat),labels=labels,enc=enc,reg=reg,Γ=pcs()).data), pcs.α, pcs.β);
            end
        end

        batch_size = Int(floor(length(L)*0.10))+2;
        mini_batchs = [];
        for _ in 1:num_steps
            mini_batch = sample(L, batch_size, replace=false);
            push!(mini_batchs, (mini_batch[1:div(batch_size,2)], mini_batch[div(batch_size,2)+1:end]));
        end
        Flux.train!((UU,LL)->loss(UU,LL; G=G,feats=masked_feats(feats,UU,L,label_feat),labels=labels,enc=enc,reg=reg,Γ=pcs()), params(enc, reg), mini_batchs, ADAM(0.001), cb=call_back(0));
        push!(rrt, accuracyFun(labels[U], pred(U, L; G=G,feats=masked_feats(feats,U,L,label_feat),labels=labels,enc=enc,reg=reg,Γ=pcs()).data));

        if inductive
            Random.seed!(seed_val);
            G, A, labels, feats = read_network(network_name1); n = nv(G);
            d = sum(sum(A), dims=1)[:];

            # Label as Feature
            label_feat && (feats = [vcat(feat,label) for (feat,label) in zip(feats,labels)]);

            update_DA!(pcs, spdiagm(0=>d), A);

            L, U = rand_split(n, ptr_inductive);
        end
        push!(rri, accuracyFun(labels[U], pred(U, L; G=G,feats=masked_feats(feats,U,L,label_feat),labels=labels,enc=enc,reg=reg,Γ=pcs()).data));

        @printf("\n pctg,    rr,    α,    β\n");
        @printf("%5.2f,    %5.2f,    %s,    %5.2f\n", ptr, rri[end], pcs.α, pcs.β);
    end
    @printf("acc:    %5.2f ± %5.2f\n", mean(rrt), std(rrt));
end

# h = plot(framestyle=:box, xlim=(0.0, 1.0), ylim=(-0.05, 1.05), legend=:right);
# plot!(h, ptrs, rr, color="blue", linestyle=:solid, label="accuracy");
# display(h);
# savefig(h, (!inductive ? network_name0 : network_name1) * ".pdf");
