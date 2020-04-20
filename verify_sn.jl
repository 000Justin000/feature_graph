using LinearAlgebra;
using Plots;
using Distributions;
using SpecialFunctions;

ϕ(x) = exp(-0.5*sum(x.^2.0)) / (2π)^(length(x)/2.0);
Φ(x) = 0.5 * (1.0 + erf(x/√2));

μ = [0.20, -0.20];
η = [1.75, -1.75];
σ = [2.11,  2.11];
N = 1000;

rg1 = -5:1:5;
rg2 = -5:1:5;
f(x) = ϕ((x.-μ)./σ)/prod(σ) * 2*Φ(dot(η, (x.-μ)./σ));

function dir_sample()
    ρ = (σ .* η) ./ sqrt(1 + sum(η.^2));

    CM = diagm(0=>σ.^2) - ρ * ρ';

    return μ .+ ρ .* abs(randn()) + cholesky(CM).L * randn(length(μ));
end

function rej_sample()
    g0 = MvNormal(μ, σ);
    while true
        x = rand(g0);
        if rand() < Φ(dot(η, (x.-μ)./σ))
            return x;
        end
    end
end

h = plot(size=(550,500));
contour!(h, rg1, rg2, (c1,c2) -> f([c1,c2]), label="true density");
rej_dat = hcat([rej_sample() for _ in 1:N]...);
scatter!(h, rej_dat[1,:], rej_dat[2,:], markersize=2.5, markerstrokewidth=0.0, color=2, label="reject sampling");
dir_dat = hcat([dir_sample() for _ in 1:N]...);
scatter!(h, dir_dat[1,:], dir_dat[2,:], markersize=2.5, markerstrokewidth=0.0, color=1, label="direct sampling");
display(h);
