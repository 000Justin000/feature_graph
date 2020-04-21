using LinearAlgebra;
using CSV;

include("utils.jl")

accuracies = [];

p = 3;
for i in 0:9
    α0 = CSV.read(ARGS[1]*string(i), delim=" ", ignorerepeated=true, header=0)[1,2:div(p*(p+1),2)+1] |> collect |> xx->[parse(Float64,x[1:end-1]) for x in xx];
    α1 = CSV.read(ARGS[1]*string(i), delim=" ", ignorerepeated=true, header=0)[end,2:div(p*(p+1),2)+1] |> collect |> xx->[parse(Float64,x[1:end-1]) for x in xx];

    push!(accuracies, 1.0 - sum((α1 .- α0) .^ 2.0) / (p/12 + p*(p-1)/2));
end

println(mean(accuracies));
println(accuracies);
