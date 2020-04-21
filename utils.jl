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

function R2(y, y_)
    """
    Args:
         y: true labels
        y_: predicted labels

    Return:
        coefficients of determination
    """

    return 1.0 - sum((y_ .- y).^2.0) / sum((y .- mean(y)).^2.0);
end

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
