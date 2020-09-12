using Juno;
using Statistics;
using Random;
using Printf;
using MLBase: roc, f1score;
using LightGraphs;
using LinearAlgebra;
using SparseArrays;
import Flux: train!;

eye(n) = diagm(0=>ones(n));
speye(n) = spdiagm(0=>ones(n));
A2D(A) = spdiagm(0=>sum(A,dims=1)[:]);
normalized_laplacian(G) = spdiagm(0=>degree(G).^-0.5) * laplacian_matrix(G) * spdiagm(0=>degree(G).^-0.5);

function rand_split(n, ptr)
    """
    Args:
         n: total number of data points
       ptr: percentage of training data

    Returns:
         L: indices for training data points
         U: indices for testing data points
    """

    randid = randperm(n);
    ll = Int64(ceil(ptr*n));

    L = randid[1:ll];
    U = randid[ll+1:end];

    return L, U;
end

function array2str(arr)
    """
    Args:
       arr: array of data
       fmt: format string
    Return:
       string representation of the array
    """

    (typeof(arr[1]) <: String) || (arr = map(x->@sprintf("%10.3f", x), arr));
    return join(arr, ", ");
end

function R2(y_, y)
    """
    Args:
        y_: predicted labels
         y: true labels

    Return:
        coefficients of determination
    """
    @assert ((ndims(y_) == ndims(y) == 1) || (size(y_,1) == size(y,1) == 1)) "unexpected input size"

    return 1.0 - sum((y_[:] .- y[:]).^2.0) / sum((y[:] .- mean(y[:])).^2.0);
end

function probmax(y_, y)
    """
    Args:
        y_: predicted probabilities
         y: true labels in one-hot encoding

    Return:
        accuracy
    """
    @assert (ndims(y_) == ndims(y) == 2) "unexpected input size"

    return sum(y[argmax(y_; dims=1)]) / size(y,2);
end

function detection_f1(y_, y)
    """
    Args:
        y_: predicted probabilities
         y: true labels in one-hot encoding

    Return:
        accuracy
    """
    @assert (ndims(y_) == ndims(y) == 2) "unexpected input size"
    @assert size(y_,1) == 2

    l  = [argmax(y[:,i]) for i in 1:size(y,2)];
    l_ = [argmax(y_[:,i]) for i in 1:size(y_,2)];

    tp = sum((l .== 2) .& (l_ .== 2));
    fp = sum((l .== 1) .& (l_ .== 2));
    tn = sum((l .== 1) .& (l_ .== 1));
    fn = sum((l .== 2) .& (l_ .== 1));

    println(tp);
    println(fp);
    println(tn);
    println(fn);

    precision = tp / (tp + fp);
    recall = tp / (tp + fn);

    return sqrt(precision * recall);
end

function expansion(m, ids)
    """
    Args:
         m: overall dimension
       ids: a length m_ vector with indices indicating location

    Returns:
         Ψ: a m x m_ matrix that expand a vector of dimension m_ to a vector of dimension m
    """

    m_ = length(ids);

    II = Vector{Int}();
    JJ = Vector{Int}();
    VV = Vector{Float64}();

    for (i,id) in enumerate(ids)
        push!(II, id);
        push!(JJ, i);
        push!(VV, 1.0);
    end

    return sparse(II, JJ, VV, m, m_);
end

# function interpolate(L, xL; Γ)
#     """
#     Args:
#          L: indices of observation
#         xL: observation
#          Γ: label propagation matrix
#
#     Returns:
#          x: interpolated value over entire graph
#     """
#
#     n = size(Γ,1);
#     U = setdiff(1:n, L);
#     xU = cg(Γ[U,U], -Γ[U,L]*xL);
#
#     x = expansion(n,L) * xL + expansion(n,U) * xU;
#
#     return x;
# end

function reset_grad!(xs...)
    for x in xs
        x.grad .= 0;
    end
end

function train!(loss, θs::Vector, mini_batches::Vector, opts::Vector; start_opts=zeros(Int,length(opts)), cb=()->(), cb_skip=1)
    """
    extend training method to allow using different optimizers for different parameters
    """

    ps = Params(vcat(collect.(θs)...));
    for (i,mini_batch) in enumerate(mini_batches)
        gs = gradient(ps) do
            loss(mini_batch...);
        end

        for (θ,opt,start_opt) in zip(θs,opts,start_opts)
            (i > start_opt) && update!(opt, θ, gs);
        end

        (i % cb_skip == 0) && cb();
    end
end
