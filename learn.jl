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
n_step = 5000;

# note G is the entity graph, A is the adjacency matrices for the graphical model
G, A, Y, X, s, d = prepare_data(dataset; N=N, p1=0, p2=3, s=[2,2,2], d=[1,1,1]);

