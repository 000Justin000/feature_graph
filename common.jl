function getA(G, p; interaction_list=[(i,j) for i in 1:p for j in i+1:p])
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
    for (i,j) in interaction_list
        (j>i) && push!(A, kron(spdiagm(0=>ones(n)), sparse([i,j], [j,i], [1.0,1.0], p, p)));
    end

    return A;
end

A2D(A) = spdiagm(0=>sum(A,dims=1)[:]);

tsctc(A, B) = reshape(A * reshape(B, (size(B,1), :)), (size(A,1), size(B)[2:end]...));

function get_ssff(d)
    ll = vcat(0, cumsum(d));
    ss = [ll[i]+1 for i in 1:length(d)];
    ff = [ll[i+1] for i in 1:length(d)];

    return ss, ff;
end

# parameter used as decoder for synthetic data
function init_W_(s, d; scale=5.0)
    if (s == 2 && d == 1)
        return reshape([-1.0, 1.0] * scale, (2,1));
    elseif (s == d)
        return diagm(0 => ones(s)) * scale;
    else
        error("unexpected (s,d) pair");
    end
end
init_b_(s) = zeros(s);

function prepare_data(dataset; N=1, p1=1, p2=1, s=[2], d=[1])
    if ((options = match(r"synthetic_([a-z]+)", dataset)) != nothing)
        if options[1] == "tiny"
            G = complete_graph(2);
        elseif options[1] == "small"
            G = watts_strogatz(10, 4, 0.3);
        elseif options[1] == "medium"
            G = watts_strogatz(500, 6, 0.3);
        elseif options[1] == "large"
            G = watts_strogatz(3000, 6, 0.3);
        else
            error("unexpected size option");
        end

        p = p1 + sum(d);
        n = nv(G);
        A = getA(G, p);
        D = A2D.(A);

        α0 = vcat(randn(p), randn(length(A)-p));
        β0 = 1.0;
        @printf("α0: %s,    β0: %10.3f\n", array2str(α0), β0);

        cr = [p1+ss_:p1+ff_ for (ss_,ff_) in zip(get_ssff(d)...)];

        CM0 = inv(Array(getΓ(α0, β0; A=A)));
        CM = (CM0 + CM0')/2.0;
        g = MvNormal(CM);
        YZ = cat([reshape(rand(g), (p,n)) for _ in 1:N]..., dims=3);
        Y = YZ[1:p1,:,:];
        Z = [YZ[cr_,:,:] for cr_ in cr];

        W0 = [param(init_W_(s[i], d[i])) for i in 1:p2];
        b0 = [param(init_b_(s[i])) for i in 1:p2];

        logPX(Z) = [logsoftmax(tsctc(W_,Z_) .+ repeat(b_,1,n,size(Z_,3)), dims=1) for (W_,b_,Z_) in zip(W0,b0,Z)];

        X = [begin
                x = zeros(size(logPX_));
                for j in 1:size(logPX_,2)
                    for k in 1:size(logPX_,3)
                        x[sample(Weights(exp.(logPX_[:,j,k]))),j,k] = 1;
                    end
                end
                x;
             end for logPX_ in logPX(Z)];
    elseif (match(r"county_election_([0-9]+)", dataset) != nothing)
        G, _, labels, feats = read_network(dataset);
        for i in vertices(G)
            rem_edge!(G, i,i);
        end

        n = nv(G);
        p1, p2 = length(feats[1]) + length(labels[1]), 0;
        s, d = Int[], Int[];
        p = p1 + sum(d);
        A = getA(G, p);

        Y = unsqueeze(hcat([vcat(feat,label) for (feat,label) in zip(feats,labels)]...), 3);
        X = [];
    elseif (match(r"cora_true_([0-9]+)", dataset) != nothing)
        G, _, labels, feats = read_network(dataset);
        for i in vertices(G)
            rem_edge!(G, i,i);
        end

        n = nv(G);
        p1, p2 = length(feats[1]), 1;
        s, d = Int[7], Int[7];
        p = p1 + sum(d);
        A = getA(G, p; interaction_list=vcat([(i,j) for i in 1:p1 for j in p1+1:p], [(i,j) for i in p1+1:p for j in i+1:p]));

        Y = unsqueeze(hcat([feat for feat in feats]...), 3);
        X = [unsqueeze(hcat([eye(7)[:,label] for label in labels]...), 3)];
    elseif (match(r"cora_false_([0-9]+)", dataset) != nothing)
        G, _, labels, feats = read_network(dataset);
        for i in vertices(G)
            rem_edge!(G, i,i);
        end

        n = nv(G);
        p1, p2 = 0, length(feats[1]) + 1;
        s, d = vcat([2 for _ in 1:p2-1], 7), vcat([1 for _ in 1:p2-1], 7);
        p = p1 + sum(d);
        A = getA(G, p);

        Y = zeros(0, n, 1);
        X = vcat([unsqueeze(hcat([eye(2)[:,feat[i]+1] for feat in feats]...), 3) for i in 1:p2], unsqueeze(hcat([eye(7)[:,label] for label in labels]...), 3));
    end

    return G, A, Y, X, s, d;
end
