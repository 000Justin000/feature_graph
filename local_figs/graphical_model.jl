using Plots; pyplot();
using LaTeXStrings;

circleShape(x, y, r) = (θ = LinRange(0, 2π, 500); Shape(x .+ r*sin.(θ), y .+ r*cos.(θ)));
squareShape(x, y, s) = Shape(x .+ [0.0,0.5,0.5,-0.5,-0.5,0.0]*s, y .+ [-0.5,-0.5,0.5,0.5,-0.5,-0.5]*s);

rr, ss, sn, dd, tt = 1.00, 0.50, 0.23, 0.50, 0.60;

b1, b2 = [0.00, 0.00], [15.00, 0.00];

x1, x2, x3 = [-3.50, -2.00],    [ 0.00,  0.00],    [ 3.50, -2.00];
d1, d2, d3 = [ 0.00,  1.00]*dd, [ 0.00,  0.00]*dd, [ 0.00, -1.00]*dd;
t1, t2, t3 = [ 0.00,  1.00]*tt, [ 0.00,  0.00]*tt, [ 0.00, -1.00]*tt;

dsp1 = [ 1.50, 0.00]
dsp2 = [ 0.00, 0.00]
dsp3 = [ 0.35, 0.00]

h = plot(size=(1000, 250), framestyle=:none);

#----------------------------------------------------------------------
# attribute graph
#----------------------------------------------------------------------
plot!(h, collect.(zip(b1 .+ x1, b1 .+ x2))..., linewidth=3.0, color=:gray);
plot!(h, collect.(zip(b1 .+ x2, b1 .+ x3))..., linewidth=3.0, color=:gray);
#
plot!(h, circleShape((b1 .+ x1)..., rr), seriestype=[:shape], linewidth=1.5, color=:white, linecolor=:black, linealpha=1.0, legend=false, fillalpha=1.0, aspect_ratio=1.0); annotate!(h, (b1 .+ x1 .+ dsp1)..., text(L"v_{1}", :black, :center, 21));
plot!(h, circleShape((b1 .+ x2)..., rr), seriestype=[:shape], linewidth=1.5, color=:white, linecolor=:black, linealpha=1.0, legend=false, fillalpha=1.0, aspect_ratio=1.0); annotate!(h, (b1 .+ x2 .+ dsp1)..., text(L"v_{2}", :black, :center, 21));
plot!(h, circleShape((b1 .+ x3)..., rr), seriestype=[:shape], linewidth=1.5, color=:white, linecolor=:black, linealpha=1.0, legend=false, fillalpha=1.0, aspect_ratio=1.0); annotate!(h, (b1 .+ x3 .+ dsp1)..., text(L"v_{3}", :black, :center, 21));
#
plot!(h, squareShape((b1 .+ x1 .+ d1)..., ss), seriestype=[:shape], linewidth=0.5, color=1, linecolor=1, legend=false, fillalpha=1.0, aspect_ratio=1.0);
plot!(h, squareShape((b1 .+ x2 .+ d1)..., ss), seriestype=[:shape], linewidth=0.5, color=1, linecolor=1, legend=false, fillalpha=1.0, aspect_ratio=1.0);
plot!(h, squareShape((b1 .+ x3 .+ d1)..., ss), seriestype=[:shape], linewidth=0.5, color=1, linecolor=1, legend=false, fillalpha=1.0, aspect_ratio=1.0);
#
plot!(h, squareShape((b1 .+ x1 .+ d2)..., ss), seriestype=[:shape], linewidth=0.5, color=2, linecolor=2, legend=false, fillalpha=1.0, aspect_ratio=1.0);
plot!(h, squareShape((b1 .+ x2 .+ d2)..., ss), seriestype=[:shape], linewidth=0.5, color=2, linecolor=2, legend=false, fillalpha=1.0, aspect_ratio=1.0);
plot!(h, squareShape((b1 .+ x3 .+ d2)..., ss), seriestype=[:shape], linewidth=0.5, color=2, linecolor=2, legend=false, fillalpha=1.0, aspect_ratio=1.0);
#
plot!(h, squareShape((b1 .+ x1 .+ d3)..., ss), seriestype=[:shape], linewidth=0.5, color=3, linecolor=3, legend=false, fillalpha=1.0, aspect_ratio=1.0);
plot!(h, squareShape((b1 .+ x2 .+ d3)..., ss), seriestype=[:shape], linewidth=0.5, color=3, linecolor=3, legend=false, fillalpha=1.0, aspect_ratio=1.0);
plot!(h, squareShape((b1 .+ x3 .+ d3)..., ss), seriestype=[:shape], linewidth=0.5, color=3, linecolor=3, legend=false, fillalpha=1.0, aspect_ratio=1.0);
#----------------------------------------------------------------------

#----------------------------------------------------------------------
# graphical model
#----------------------------------------------------------------------
plot!(h, collect.(zip(b2 .+ x1 .+ t1 * 1.5, b2 .+ x1 .+ t3 * 1.5))..., linewidth=3.0, color=:gray);
plot!(h, collect.(zip(b2 .+ x2 .+ t1 * 1.5, b2 .+ x2 .+ t3 * 1.5))..., linewidth=3.0, color=:gray);
plot!(h, collect.(zip(b2 .+ x3 .+ t1 * 1.5, b2 .+ x3 .+ t3 * 1.5))..., linewidth=3.0, color=:gray);
#
plot!(h, collect.(zip(b2 .+ x1 .+ t1, b2 .+ x2 .+ t1))..., linewidth=0.5, color=1);
plot!(h, collect.(zip(b2 .+ x1 .+ t2, b2 .+ x2 .+ t2))..., linewidth=0.5, color=2);
plot!(h, collect.(zip(b2 .+ x1 .+ t3, b2 .+ x2 .+ t3))..., linewidth=0.5, color=3);
#
plot!(h, collect.(zip(b2 .+ x2 .+ t1, b2 .+ x3 .+ t1))..., linewidth=0.5, color=1);
plot!(h, collect.(zip(b2 .+ x2 .+ t2, b2 .+ x3 .+ t2))..., linewidth=0.5, color=2);
plot!(h, collect.(zip(b2 .+ x2 .+ t3, b2 .+ x3 .+ t3))..., linewidth=0.5, color=3);
#
plot!(h, circleShape((b2 .+ x1)..., rr), seriestype=[:shape], linewidth=1.5, color=:white, linestyle=:dash, linecolor=:black, linealpha=0.1, legend=false, fillalpha=0.0, aspect_ratio=1.0);
plot!(h, circleShape((b2 .+ x2)..., rr), seriestype=[:shape], linewidth=1.5, color=:white, linestyle=:dash, linecolor=:black, linealpha=0.1, legend=false, fillalpha=0.0, aspect_ratio=1.0);
plot!(h, circleShape((b2 .+ x3)..., rr), seriestype=[:shape], linewidth=1.5, color=:white, linestyle=:dash, linecolor=:black, linealpha=0.1, legend=false, fillalpha=0.0, aspect_ratio=1.0);
#
plot!(h, circleShape((b2 .+ x1 .+ t1)..., sn), seriestype=[:shape], linewidth=0.5, color=1, linecolor=:black, legend=false, fillalpha=1.0, aspect_ratio=1.0); annotate!(h, (b2 .+ x1 .+ 1.4 * t1 .+ dsp3)..., text(L"x_{11}", :black, :left, 14));
plot!(h, circleShape((b2 .+ x2 .+ t1)..., sn), seriestype=[:shape], linewidth=0.5, color=1, linecolor=:black, legend=false, fillalpha=1.0, aspect_ratio=1.0); annotate!(h, (b2 .+ x2 .+ 1.4 * t1 .+ dsp3)..., text(L"x_{21}", :black, :left, 14));
plot!(h, circleShape((b2 .+ x3 .+ t1)..., sn), seriestype=[:shape], linewidth=0.5, color=1, linecolor=:black, legend=false, fillalpha=1.0, aspect_ratio=1.0); annotate!(h, (b2 .+ x3 .+ 1.4 * t1 .+ dsp3)..., text(L"x_{31}", :black, :left, 14));
#
plot!(h, circleShape((b2 .+ x1 .+ t2)..., sn), seriestype=[:shape], linewidth=0.5, color=2, linecolor=:black, legend=false, fillalpha=1.0, aspect_ratio=1.0); annotate!(h, (b2 .+ x1 .+ 1.0 * t2 .+ dsp3)..., text(L"x_{12}", :black, :left, 14));
plot!(h, circleShape((b2 .+ x2 .+ t2)..., sn), seriestype=[:shape], linewidth=0.5, color=2, linecolor=:black, legend=false, fillalpha=1.0, aspect_ratio=1.0); annotate!(h, (b2 .+ x2 .+ 1.0 * t2 .+ dsp3)..., text(L"x_{22}", :black, :left, 14));
plot!(h, circleShape((b2 .+ x3 .+ t2)..., sn), seriestype=[:shape], linewidth=0.5, color=2, linecolor=:black, legend=false, fillalpha=1.0, aspect_ratio=1.0); annotate!(h, (b2 .+ x3 .+ 1.0 * t2 .+ dsp3)..., text(L"x_{32}", :black, :left, 14));
#
plot!(h, circleShape((b2 .+ x1 .+ t3)..., sn), seriestype=[:shape], linewidth=0.5, color=3, linecolor=:black, legend=false, fillalpha=1.0, aspect_ratio=1.0); annotate!(h, (b2 .+ x1 .+ 1.4 * t3 .+ dsp3)..., text(L"y_{1}",  :black, :left, 14));
plot!(h, circleShape((b2 .+ x2 .+ t3)..., sn), seriestype=[:shape], linewidth=0.5, color=3, linecolor=:black, legend=false, fillalpha=1.0, aspect_ratio=1.0); annotate!(h, (b2 .+ x2 .+ 1.4 * t3 .+ dsp3)..., text(L"y_{2}",  :black, :left, 14));
plot!(h, circleShape((b2 .+ x3 .+ t3)..., sn), seriestype=[:shape], linewidth=0.5, color=3, linecolor=:black, legend=false, fillalpha=1.0, aspect_ratio=1.0); annotate!(h, (b2 .+ x3 .+ 1.4 * t3 .+ dsp3)..., text(L"y_{3}",  :black, :left, 14));
#----------------------------------------------------------------------

annotate!(h, ((b1 .+ b2) / 2.0 .+ x2*0.0)..., text(L"E[\mathbf{y}_{U}|\mathbf{X}] = (\mathbf{X} \beta)_{U}", :black, :left, 10))
annotate!(h, ((b1 .+ b2) / 2.0 .+ x1*0.3)..., text(L"E[\mathbf{y}_{U}|\mathbf{y}_{L}] = -(\mathbf{I} + \omega \mathcal{L})_{UU}^{-1} (\mathbf{I} + \omega \mathcal{L})_{UL} \mathbf{y}_{L}", :black, :left, 10))
annotate!(h, ((b1 .+ b2) / 2.0 .+ x1*0.6)..., text(L"E[\mathbf{y}_{U}|\mathbf{X}] = [(\mathbf{I} + \omega \mathcal{L})^{-1} \mathbf{X} \beta]_{U}", :black, :left, 10))
annotate!(h, ((b1 .+ b2) / 2.0 .- x3*0.3)..., text(L"E[\mathbf{y}_{U}|\mathbf{X}, \mathbf{y}_{L}] =", :black, :left, 10))

savefig(h, "graphical_model.svg");
