using Random;
using Printf;
using MLBase: roc, f1score;
using LightGraphs;

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

function getA(G, p; interaction_list=1:p)
    """
    Given a graph, generate the graphical model that has every vertex mapped to
    p vertices, with p of them representing features

    Args:
       G: LightGraph Object
       p: number of features
    Return:
       A: an array of adjacency matrices for the graphical model
    """
    n = nv(G); A0 = adjacency_matrix(G);

    A = Vector{SparseMatrixCSC}();

    # connections among corresponding features on different vertices
    for i in 1:p
        push!(A, kron(A0, sparse([i], [i], [1.0], p, p)));
    end

    # connections among different features on same vertices
    for i in 1:p
        for j in interaction_list
            (j > i) && push!(A, kron(spdiagm(0=>ones(n)), sparse([i,j], [j,i], [1.0,1.0], p, p)));
        end
    end

    return A;
end

A2D(A) = spdiagm(0=>sum(A,dims=1)[:]);

function reset_grad!(xs...)
    for x in xs
        x.grad .= 0;
    end
end
