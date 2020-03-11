using Flux;
using LinearAlgebra;
using SparseArrays;
using LightGraphs;
using Plots; pyplot();
using LaTeXStrings;
using Distributions;

include("utils.jl");
include("read_network.jl");

p = 3;
G = complete_graph(2);
n = nv(G);
A = Array.(getA(G, p));
D = Array.(A2D.(A));

φ = param(zeros(4));
getα() = vcat(ones(p)*φ[1], ones(div(p*(p-1),2))*φ[2], ones(p)*φ[3]);
getβ() = exp(φ[4]);

function getGamma(α, β)
    Gamma = β * I + β * sum([abs(α_)*D_ - α_*A_ for (α_,D_,A_) in zip(α,D,A)]);

    return Gamma;
end

function Omega(X::Vector{Vector{Float64}})
    Gamma = getGamma(getα(), getβ());

    return logdet(Array(Gamma)) - mean([x'*Gamma*x for x in X]);
end

function Omega(CM::Array{Float64,2})
    Gamma = getGamma(getα(), getβ());

    return logdet(Array(Gamma)) - sum(CM .* Gamma);
end

CM0 = inv(Array(getGamma(ones(div(p*(p+3),2))*3.0, exp(3.0))));
CM = (CM0 + CM0')/2.0;
g = MvNormal(CM);
X = [rand(g) for _ in 1:1000];

println(φ);
Flux.train!(X -> -Omega(X), Flux.params(φ), [(sample(X,10),) for _ in 1:10000], ADAM(0.1))
println(φ);
