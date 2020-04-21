using LinearAlgebra;
using CSV;

include("utils.jl")

accuracies = [];

p = 3;
αα = vcat(ones(p), zeros(div(p*(p-1),2)));
for i in 0:9
    α0 = CSV.read("output"*string(i), delim=" ", ignorerepeated=true, header=0)[1,2:div(p*(p+1),2)+1] |> collect |> xx->[parse(Float64,x[1:end-1]) for x in xx];
    α1 = CSV.read("output"*string(i), delim=" ", ignorerepeated=true, header=0)[end,2:div(p*(p+1),2)+1] |> collect |> xx->[parse(Float64,x[1:end-1]) for x in xx];

    push!(accuracies, 1.0 - mean((α1 .- α0) .^ 2.0));
end

print(accuracies);
