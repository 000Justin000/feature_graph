using Flux;
using Zygote;
using LinearAlgebra;
using SparseArrays;
using LightGraphs;
using Plots; pyplot();
using LaTeXStrings;

include("utils.jl");
include("read_network.jl");

p = 8;
t = 512; k = 32;
# G, A, labels, feats = read_network("county_election_2016");
G = complete_graph(32);
# G = watts_strogatz(3,2,0.5);
n = nv(G);
AA = getAA(G, p);

# φ = zeros(p + div(p*(p-1),2) + p + 1);
# getα() = tanh.(φ[1:end-1]);
# getβ() = exp(φ[end]);

φ = zeros(3);
getα() = vcat(ones(p)*φ[1], ones(div(p*(p-1),2))*φ[2], ones(p)*φ[3]);
getβ() = 1.0;

L0, U0 = rand_split(n, 0.5);

function getGamma(α, β=nothing; AA)
    function Deg(A)
        d = sum(A, dims=1)[:];
        return spdiagm(0=>d);
    end

    DD = Deg.(AA);
    Gm = I + sum(abs.(α) .* DD) - sum(α .* AA);
    
    if !isnothing(β)
        Gamma = β * Gm;
    else
        Gamma = Gm * exp(-logdet(Array(Gm))/size(Gm,1));
    end

    return Gamma;
end

function logdet_gt(L, U; msg="")
    Gamma = getGamma(getα(); AA=AA);
    result = logdet(Array(Gamma[setdiff(1:n*2*p,L),setdiff(1:n*2*p,L)])) - 
             logdet(Array(Gamma[setdiff(1:n*2*p,L,U),setdiff(1:n*2*p,L,U)]));
    return result;
end

U = (U0.-1)*2*p .+ (p+1);

α1 = range(-10.0, 10.0; length=30);
α2 = range(-10.0, 10.0; length=30);

V_LP  = zeros(length(α1), length(α2));
V_GNN = zeros(length(α1), length(α2));

for i in 1:length(α1)
    for j in 1:length(α2)
        global φ = [α1[i], α2[j], 1.0];
        V0 = (-logdet_gt([],U));

        # label propagation
        L_LP = (L0.-1)*2*p .+ (p+1);
        V_LP[i,j] = (-logdet_gt(L_LP,U)) - V0;

        # GNN
        L_GNN = vcat([(collect(1:n).-1)*2*p .+ (p+i) for i in 2:p]...);
        V_GNN[i,j] = (-logdet_gt(L_GNN,U)) - V0;
    end
end

function make_plot(α1, α2, V, fname="tmp", title="volume")
    h = plot(framestyle=:box, xlabel=L"\alpha_{1}", ylabel=L"\alpha_{2}", title=title, size=(500,350));
    contour!(h, α1, α2, V', fill=true, color=ColorGradient([:green,:white,:red]), clims=(-1.0,1.0).*maximum(abs.(V)));
    savefig(h, "figs/"*fname*".png");
end

make_plot(α1, α2, V_LP,       @sprintf("%02d_%02d_LP", n, p),  @sprintf("volume LP (n = %d, p = %d)", n, p));
make_plot(α1, α2, V_GNN,      @sprintf("%02d_%02d_GNN", n, p), @sprintf("volume GNN (n = %d, p = %d)", n, p));
make_plot(α1, α2, V_LP-V_GNN, @sprintf("%02d_%02d_GAP", n, p), @sprintf("volume LP - GNN (n = %d, p = %d)", n, p));
