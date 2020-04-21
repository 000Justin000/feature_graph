using LinearAlgebra;
using CSV;

include("utils.jl")

accuracies0 = [];
accuracies1 = [];
accuracies2 = [];

p = 3;
for i in 0:9
    α0 = CSV.read(ARGS[1]*string(i), delim=" ", ignorerepeated=true, header=0)[1,2:div(p*(p+1),2)+1] |> collect |> xx->[parse(Float64,x[1:end-1]) for x in xx];
    α1 = CSV.read(ARGS[1]*string(i), delim=" ", ignorerepeated=true, header=0)[end,2:div(p*(p+1),2)+1] |> collect |> xx->[parse(Float64,x[1:end-1]) for x in xx];

    push!(accuracies0, sum(sign.(α0) .== sign.(α1)) / length(α0));
    push!(accuracies1, cor(α0, α1));
    push!(accuracies2, 1.0 - mean((α1 .- α0) .^ 2.0));
end

println(mean(accuracies0));
println(accuracies0);
println(mean(accuracies1));
println(accuracies1);
println(mean(accuracies2));
println(accuracies2);
