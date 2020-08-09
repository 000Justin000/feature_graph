using Plots;

lines = readlines(ARGS[1]);

xx = 0.05:0.05:0.60;
ac = Dict{String,Vector{Float64}}();
mi = Dict{String,Vector{Float64}}();

for line in lines
    if (p = match(r"[ ]*([a-zA-Z0-9_]+) AC:[ ]+([-0-9\.]+)", line)) != nothing
        !haskey(ac, p[1]) && (ac[p[1]] = Vector{Float64}());
        push!(ac[p[1]], parse(Float64, p[2]))
    elseif (p = match(r"[ ]*([a-zA-Z0-9_]+) MI:[ ]+([-0-9\.]+)", line)) != nothing
        !haskey(mi, p[1]) && (mi[p[1]] = Vector{Float64}());
        push!(mi[p[1]], parse(Float64, p[2]))
    end
end

h = plot(size=(550, 500), xlim=(-0.02, 0.62), ylim=(-0.02, 1.02), framestyle=:box, legend=:none);

algos = ["LP", "LR", "SLR", "LR_LP", "SLR_LP", "GNN", "GNN_LP"];
for (i,algo) in enumerate(algos)
    plot!(h, xx, ac[algo], linestyle=:solid, color=i, label="AC "*algo);
    scatter!(h, xx, ac[algo], linestyle=:dash, color=i, label="");
end

if length(mi) != 0
    max_mi = maximum(vcat(values(mi)...));
    mi_const = min(max_mi, 1.0);

    for (i,algo) in enumerate(algos)
        if haskey(mi, algo) 
            plot!(h, xx,    mi[algo]/max_mi*mi_const, linestyle=:dash, color=i, label="MI "*algo);
            scatter!(h, xx, mi[algo]/max_mi*mi_const, linestyle=:dash, color=i, label="");
        end
    end
end

savefig(h, ARGS[1]*"_result.svg");
