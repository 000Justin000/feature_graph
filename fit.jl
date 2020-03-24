using Flux;
using Flux: train!, throttle;
using LinearAlgebra;
using SparseArrays;
using LightGraphs;
using Plots; pyplot();
using LaTeXStrings;
using Distributions;

include("utils.jl");
include("kernels.jl");
include("read_network.jl");

G, _, labels, feats = read_network("county_election_2016");
for i in vertices(G)
    rem_edge!(G, i,i);
end

p = length(feats[1]) + length(labels[1]);
t, k = 128, 32;
n = nv(G);
A = getA(G, p);
D = A2D.(A);

X = [vcat([vcat(feat,label) for (feat,label) in zip(feats,labels)]...)];

# α0 = vcat(randn(p), randn(div(p*(p-1),2)));
# β0 = exp(randn());
# CM0 = inv(Array(getΓ(α0, β0; A=A)));
# CM = (CM0 + CM0')/2.0;
# g = MvNormal(CM);
# X = [rand(g) for _ in 1:1];

φ = param(zeros(div(p*(p+1),2)+1));
getα() = vcat(φ[1:p], φ[p+1:end-1]);
getβ() = exp(φ[end]);
print_params() = @printf("α:  %s,    β:  %10.3f\n", array2str(getα()), getβ());

function loss(xx, L; getα=getα, getβ=getβ)
    rr = [x for x in xx];
    V = collect(1:size(A[1],1));
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
α1 = CSV.read("output", delim=" ", ignorerepeated=true, header=0)[end,2:div(p*(p+1),2)+1] |> collect |> xx->[parse(Float64,x[1:end-1]) for x in xx];
β1 = CSV.read("output", delim=" ", ignorerepeated=true, header=0)[end,end];
FIDX(fidx, V=vertices(G)) = [(i-1)*p+j for i in V for j in fidx];
L, U = rand_split(nv(G), 0.6);

function print_vol(lidx=[7], fidx=[1,2,3,4,5,6])
    obsl = FIDX(lidx, L); obsf = FIDX(fidx, vertices(G)); trgl = FIDX(lidx, U);
    vol_base() = (-logdetΓ(param(α1), param(β1); A=A, P=vcat(obsf,obsl,trgl), t=t, k=k)) - (-logdetΓ(param(α1), param(β1); A=A, P=vcat(obsf,obsl), t=t, k=k));
    vol_lp()  = (-logdetΓ(param(α1), param(β1); A=A, P=vcat(obsf,trgl), t=t, k=k)) - (-logdetΓ(param(α1), param(β1); A=A, P=obsf, t=t, k=k));
    vol_gnn() = (-logdetΓ(param(α1), param(β1); A=A, P=vcat(obsl,trgl), t=t, k=k)) - (-logdetΓ(param(α1), param(β1); A=A, P=obsl, t=t, k=k));
    vol_cgnn() = (-logdetΓ(param(α1), param(β1); A=A, P=trgl, t=t, k=k));

    @printf("base:    %10.4f\n", mean([vol_base() for _ in 1:10]));
    @printf("lp:      %10.4f\n", mean([vol_lp() for _ in 1:10]));
    @printf("gnn:     %10.4f\n", mean([vol_gnn() for _ in 1:10]));
    @printf("cgnn:    %10.4f\n", mean([vol_cgnn() for _ in 1:10]));
end
