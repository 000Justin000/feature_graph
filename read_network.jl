using LightGraphs;
using IsingLite: spingrid, metropolis!;
using StatsBase;
using Statistics;
using DelimitedFiles;
using DataFrames;
using CSV;
using JSON;
using MAT;
using Flux;
using Arpack;
using LinearAlgebra;
using MultivariateStats;

max_normalize(x) = maximum(abs.(x)) == 0 ? x : x/maximum(abs.(x));
std_normalize(x) = std(x) == 0 ? zeros(length(x)) : (x.-mean(x))./std(x);
int_normalize(x) = std(x) == 0 ? zeros(length(x)) : (x.-minimum(x))/(maximum(x).-minimum(x))*2 .- 1;

function parse_mean_fill(vr, normalize=false)
    vb = mean(map(x->(typeof(x)<:Union{Float64,Int} ? x : parse(Float64, replace(x, ","=>""))), filter(!ismissing, vr)));
    vv = collect(map(x->ismissing(x) ? vb : (typeof(x)<:Union{Float64,Int} ? x : parse(Float64, replace(x, ","=>""))), vr));
    if normalize
        vv = (vv .- vb) / std(vv);
    end
    return vv;
end

function simulate_ising(n, h0, J)
    g = LightGraphs.grid([n,n]);

    l = range(-1.0, 1.0, length=n);
    s = spingrid(n);
    metropolis!(s, h=h0*(l.*l'), J=J, iters=1000000, plot=false);
    y = s[:];
    f = [[l[i],l[j]] for j in 1:n for i in 1:n];

    return g, [adjacency_matrix(g)], y, f;
end

function read_county(prediction, year)
    # construct graph
    adj = CSV.read("datasets/election/adjacency.txt", header=0);
    fips2cty = Dict();
    for i in 1:size(adj,1)
        if !ismissing(adj[i,2])
            fips2cty[adj[i,2]] = adj[i,1];
        end
    end

    hh = adj[:,2];
    tt = adj[:,4];

    @assert !ismissing(hh[1]);
    for i in 2:size(hh,1)
        ismissing(hh[i]) && (hh[i] = hh[i-1]);
    end
    hh = convert(Vector{Int}, hh);

    fips = sort(unique(union(hh,tt)));
    id2num = Dict(id=>num for (num,id) in enumerate(fips));
    G = Graph(length(id2num));
    for (h,t) in zip(hh,tt)
        add_edge!(G, id2num[h], id2num[t]);
    end

    VOT = CSV.read("datasets/election/election.csv");
    ICM = CSV.read("datasets/election/income.csv");
    POP = CSV.read("datasets/election/population.csv");
    EDU = CSV.read("datasets/election/education.csv");
    UEP = CSV.read("datasets/election/unemployment.csv");

    cty = DataFrames.DataFrame((:FIPS=>fips, :County=>[fips2cty[fips_] for fips_ in fips]));
    vot = DataFrames.DataFrame((:FIPS=>VOT[:,:fips_code], :DEM=>VOT[:,Symbol("dem_", year)], :GOP=>VOT[:,Symbol("gop_", year)]));
    icm = DataFrames.DataFrame((:FIPS=>ICM[:,:FIPS], :MedianIncome=>ICM[:,Symbol("MedianIncome", min(max(2011,year), 2018))]));
    pop = DataFrames.DataFrame((:FIPS=>POP[:,:FIPS], :MigraRate=>POP[:,Symbol("R_NET_MIG_", min(max(2011,year), 2018))],
                                                     :BirthRate=>POP[:,Symbol("R_birth_", min(max(2011,year), 2018))],
                                                     :DeathRate=>POP[:,Symbol("R_death_", min(max(2011,year), 2018))]));
    edu = DataFrames.DataFrame((:FIPS=>EDU[:,Symbol("FIPS")], :BachelorRate=>EDU[:,Symbol("BachelorRate", year)]));
    uep = DataFrames.DataFrame((:FIPS=>UEP[:,:FIPS], :UnemploymentRate=>UEP[:,Symbol("Unemployment_rate_", min(max(2007,year), 2018))]));

    jfl(df1, df2) = join(df1, df2, on=:FIPS, kind=:inner);
    dat = sort(jfl(jfl(jfl(jfl(jfl(cty, vot), icm), pop), edu), uep), [:FIPS]);
    g, ori_id = induced_subgraph(G, [id2num[id] for id in dat[:,:FIPS]]);

    # extract features and label
    dem = parse_mean_fill(dat[:,3]);
    gop = parse_mean_fill(dat[:,4]);
    ff = zeros(Float32, size(dat,1),7);
    for i in 1:6
        ff[:,i] = parse_mean_fill(dat[:,i+4], true);
    end
    ff[:,7] = parse_mean_fill((gop-dem)./(gop+dem), true);

    if prediction == "income"
        pos = 1;
    elseif prediction == "migration"
        pos = 2;
    elseif prediction == "birth"
        pos = 3;
    elseif prediction == "death"
        pos = 4;
    elseif prediction == "education"
        pos = 5;
    elseif prediction == "unemployment"
        pos = 6;
    elseif prediction == "election"
        pos = 7;
    else
        error("unexpected prediction type");
    end

    y = ff[:,pos];
    f = [vcat(ff[i,1:pos-1], ff[i,pos+1:end]) for i in 1:size(ff,1)];

    return g, [adjacency_matrix(g)], y, f;
end

function read_environment(prediction, year)
    adj = CSV.read("datasets/election/adjacency.txt", header=0);
    fips2cty = Dict();
    for i in 1:size(adj,1)
        if !ismissing(adj[i,2])
            fips2cty[adj[i,2]] = adj[i,1];
        end
    end

    hh = adj[:,2];
    tt = adj[:,4];

    @assert !ismissing(hh[1]);
    for i in 2:size(hh,1)
        ismissing(hh[i]) && (hh[i] = hh[i-1]);
    end
    hh = convert(Vector{Int}, hh);

    fips = sort(unique(union(hh,tt)));
    id2num = Dict(id=>num for (num,id) in enumerate(fips));
    G = Graph(length(id2num));
    for (h,t) in zip(hh,tt)
        add_edge!(G, id2num[h], id2num[t]);
    end

    MAT = filter(x -> x[:Year] == year, CSV.read("datasets/environment/max_air_temperature.txt"));
    LST = filter(x -> x[:Year] == year, CSV.read("datasets/environment/land_surface_temperature.txt"));
    PCP = filter(x -> x[:Year] == year, CSV.read("datasets/environment/percipitation.txt"));
    SLT = filter(x -> x[:Year] == year, CSV.read("datasets/environment/sunlight.txt"));
    FPM = filter(x -> x[:Year] == year, CSV.read("datasets/environment/fine_particulate_matter.txt"));

    cty = DataFrames.DataFrame((:FIPS=>fips, :County=>[fips2cty[fips_] for fips_ in fips]));
    mat = DataFrames.DataFrame((:FIPS=>MAT[:,Symbol("County Code")], :MAT=>MAT[:,Symbol("Avg Daily Max Air Temperature (C)")]));
    lst = DataFrames.DataFrame((:FIPS=>LST[:,Symbol("County Code")], :LST=>LST[:,Symbol("Avg Day Land Surface Temperature (C)")]));
    pcp = DataFrames.DataFrame((:FIPS=>PCP[:,Symbol("County Code")], :PCP=>PCP[:,Symbol("Avg Daily Precipitation (mm)")]));
    slt = DataFrames.DataFrame((:FIPS=>SLT[:,Symbol("County Code")], :SLT=>SLT[:,Symbol("Avg Daily Sunlight (KJ/m\xb2)")]));
    fpm = DataFrames.DataFrame((:FIPS=>FPM[:,Symbol("County Code")], :FPM=>FPM[:,Symbol("Avg Fine Particulate Matter (\xb5g/m\xb3)")]));

    jfl(df1, df2) = join(df1, df2, on=:FIPS, kind=:inner);
    dat = sort(jfl(jfl(jfl(jfl(jfl(cty, mat), lst), pcp), slt), fpm), [:FIPS]);
    g, ori_id = induced_subgraph(G, [id2num[id] for id in dat[:,:FIPS]]);

    ff = zeros(Float32, size(dat,1), 5);
    for i in 1:5
        ff[:,i] = parse_mean_fill(dat[:,i+2], true);
    end

    if prediction == "airT"
        pos = 1;
    elseif prediction == "landT"
        pos = 2;
    elseif prediction == "precipitation"
        pos = 3;
    elseif prediction == "sunlight"
        pos = 4;
    elseif prediction == "pm2.5"
        pos = 5;
    else
        error("unexpected prediction type");
    end

    y = ff[:,pos];
    f = [vcat(ff[i,1:pos-1], ff[i,pos+1:end]) for i in 1:size(ff,1)];

    return g, [adjacency_matrix(g)], y, f;
end


function read_ward(prediction, year)
    code = CSV.read("datasets/london/code.csv", header=1);
    W = CSV.read("datasets/london/W.csv", header=1)
    dat = CSV.read("datasets/london/MaycoordLondon.csv", header=1);

    info = join(code[:,2:end], dat, on=names(code)[2] => names(dat)[1], kind=:left);

    indices = (!ismissing).(info[:,2]);
    W0 = Matrix(W[:,2:end] .!== 0.0)[indices, indices];
    info0 = info[indices,:];

    function parse_mean_fill(vr, normalize=false)
        vb = mean(map(x->(typeof(x)<:Union{Float64,Int} ? x : parse(Float64, replace(x, ","=>""))), filter(!ismissing, vr)));
        vv = collect(map(x->ismissing(x) ? vb : (typeof(x)<:Union{Float64,Int} ? x : parse(Float64, replace(x, ","=>""))), vr));
        if normalize
            vv = (vv .- vb) / std(vv);
        end
        return vv;
    end

    col = [5, 6, 7, 10, 11];
    ff = zeros(Float32, size(info0,1),6);
    for i in 1:5
        ff[:,i] = parse_mean_fill(info0[:,col[i]], true);
    end

    if year == "2016"
        ff[:,6] = parse_mean_fill(log.(info0[:,3]), true);
    elseif year == "2012"
        ff[:,6] = parse_mean_fill(log.(info0[:,4]), true);
    else
        error("unexpected year");
    end

    if prediction == "edu"
        pos = 1;
    elseif prediction == "age"
        pos = 2;
    elseif prediction == "gender"
        pos = 3;
    elseif prediction == "income"
        pos = 4;
    elseif prediction == "populationsize"
        pos = 5;
    elseif prediction == "election"
        pos = 6;
    else
        error("unexpected prediction type");
    end

    y = ff[:,pos];
    f = [vcat(ff[i,1:pos-1], ff[i,pos+1:end]) for i in 1:size(ff,1)];

    return Graph(W0), [float(W0)], y, f;
end

function read_twitch(cnm, dim_reduction=false, dim_embed=8)
    #----------------------------------------------------------------------------
    feats_all = [];
    for cn in ["DE", "ENGB", "ES", "FR", "PTBR", "RU"]
        feats = JSON.parsefile("datasets/twitch/" * cn * "/musae_" * cn * "_features.json");
        append!(feats_all, values(feats));
    end
    #----------------------------------------------------------------------------
    ndim = maximum(vcat(feats_all...)) + 1;
    #----------------------------------------------------------------------------
    function feat_encode(feat_list)
        vv = zeros(Float32, ndim);
        vv[feat_list .+ 1] .= 1.0;
        return vv;
    end
    #----------------------------------------------------------------------------
    f_all = feat_encode.(feats_all);
    #----------------------------------------------------------------------------

    #----------------------------------------------------------------------------
    feats = JSON.parsefile("datasets/twitch/" * cnm * "/musae_" * cnm * "_features.json");
    id2ft = Dict(id+1=>ft for (id,ft) in zip(parse.(Int,keys(feats)), values(feats))); n = length(id2ft);
    @assert minimum(keys(id2ft)) == 1 && maximum(keys(id2ft)) == n;
    #----------------------------------------------------------------------------
    f = [feat_encode(id2ft[id]) for id in sort(collect(keys(id2ft)))];
    #----------------------------------------------------------------------------

    if dim_reduction
        U,S,V = svds(hcat(f...); nsv=dim_embed)[1];
        UU = U .* sign.(sum(U,dims=1)[:])';
        f = [UU'*f_ for f_ in f];
        fbar = mean(f);
        f = [f_ - fbar for f_ in f];
    end

    g = Graph(length(f));
    links = CSV.read("datasets/twitch/" * cnm * "/musae_" * cnm * "_edges.csv");
    for i in 1:size(links,1)
        add_edge!(g, links[i,:from]+1, links[i,:to]+1);
    end

    trgts = CSV.read("datasets/twitch/" * cnm * "/musae_" * cnm * "_target.csv");
    nid2views = Dict(zip(trgts[!,:new_id], trgts[!,:views]));
    y = std_normalize(log.([nid2views[i-1] for i in 1:nv(g)] .+ 1.0));

    return g, [adjacency_matrix(g)], y, f;
end

function read_transportation_network(network_name, net_skips, net_cols, netf_cols, flow_skips, flow_cols, V)
    dat_net  = readdlm("datasets/transportation/" * network_name * "/" * network_name * "_net.tntp",  skipstart=net_skips)[:, net_cols];
    dat_netf = readdlm("datasets/transportation/" * network_name * "/" * network_name * "_net.tntp",  skipstart=net_skips)[:, netf_cols];
    dat_flow = readdlm("datasets/transportation/" * network_name * "/" * network_name * "_flow.tntp", skipstart=flow_skips)[:, flow_cols];

    lb2id = Dict{Int,Int}(v=>i for (i,v) in enumerate(V));
    NV = length(V);

    g = DiGraph(NV);
    for i in 1:size(dat_net,1)dim_embed
        src, dst = dat_net[i,1], dat_net[i,2];
        if haskey(lb2id, src) && haskey(lb2id, dst)
            add_edge!(g, lb2id[src], lb2id[dst]);
        end
    end

    #---------------------
    # edge labels
    #---------------------
    flow_dict = Dict{Tuple{Int,Int}, Float64}();
    for i in 1:size(dat_flow,1)
        src, dst = dat_flow[i,1], dat_flow[i,2];
        if haskey(lb2id, src) && haskey(lb2id, dst)
            flow_dict[(lb2id[src], lb2id[dst])] = dat_flow[i,3];
        end
    end
    #---------------------
    y = std_normalize([flow_dict[(e.src, e.dst)] for e in edges(g)]);
    #---------------------

    #---------------------
    # edge features
    #---------------------
    netf_dict = Dict{Tuple{Int,Int}, Vector{Float32}}();
    for i in 1:size(dat_net,1)
        src, dst = dat_net[i,1], dat_net[i,2];
        if haskey(lb2id, src) && haskey(lb2id, dst)
            netf_dict[(lb2id[src], lb2id[dst])] = dat_netf[i,:];
        end
    end
    #---------------------
    ff = vcat([netf_dict[(e.src, e.dst)]' for e in edges(g)]...);
    #---------------------
    netf = zeros(Float32, size(ff));
    for i in 1:length(netf_cols)
        netf[:,i] = std_normalize(ff[:,i]);
    end
    #---------------------
    f = [netf[i,:] for i in 1:size(netf,1)];
    #---------------------

    #---------------------
    # line graph transformation
    #---------------------
    G1 = Graph(ne(g));
    G2 = Graph(ne(g));

    tuple2id = Dict((e.src, e.dst) => i for (i,e) in enumerate(edges(g)));

    for u in vertices(g)
        innbrs = inneighbors(g, u);
        outnbrs = outneighbors(g, u);

        for v in innbrs
            for w in outnbrs, dim_reduction=false
                add_edge!(G1, tuple2id[(v,u)], tuple2id[(u,w)]);
            end
        end

        for v in innbrs
            for w in innbrs
                if w > v
                    add_edge!(G2, tuple2id[(v,u)], tuple2id[(w,u)]);
                end
            end
        end

        for v in outnbrs
            for w in outnbrs
                if w > v
                    add_edge!(G2, tuple2id[(u,v)], tuple2id[(u,w)]);
                end
            end
        end
    end

    A = adjacency_matrix.([G1, G2]);

    return Graph(sum(A)), A, y, f;
end

function read_sexual(studynum)
    V0 = filter(row -> row[:STUDYNUM]==studynum && row[:SEX] in [0,1], CSV.read("datasets/icpsr_22140/DS0001/22140-0001-Data.tsv"));
    E0 = filter(row -> row[:STUDYNUM]==studynum && row[:TIETYPE] == 3, CSV.read("datasets/icpsr_22140/DS0002/22140-0002-Data.tsv"));

    @assert length(unique(V0[!,:RID])) == size(V0,1);
    G0 = Graph(size(V0,1));
    id2num = Dict(id=>num for (num,id) in enumerate(V0[!,:RID]));

    for i in 1:size(E0,1)
        if (haskey(id2num, E0[i,:ID1]) && haskey(id2num, E0[i,:ID2]))
            add_edge!(G0, id2num[E0[i,:ID1]], id2num[E0[i,:ID2]]);
        end
    end

    lcc = sort(connected_components(G0), by=cc->length(cc))[end];
    G,_ = induced_subgraph(G0, lcc)
    V = V0[lcc,:];

    y = 1.0 .- V[!,:SEX] * 2.0;

    function filter_mean_fill(x0, f, normalize=false)
        x = convert(Vector{Float32}, x0);
        m = mean(x[f.(x)]);
        x[(!f).(x)] .= m;

        normalize && (x = std_normalize(x));

        return x;
    end

    features = [];
    push!(features, hcat(map(x -> Flux.onehot(x, collect(1:5)), V[!,:RACE])...)');
    push!(features, filter_mean_fill(V[!,:BEHAV], x -> x>=0, true));

    ff = hcat(features...);
    f = [ff[i,:] for i in 1:size(ff,1)];

    return G, [adjacency_matrix(G)], y, f;
end

function read_cora(dim_reduction=false, dim_embed=8)
    cnt = CSV.read("datasets/cora/cora.content", header=0);
    cls = sort(unique(cnt[:,end]));
    cid2num = Dict(id=>num for (num,id) in enumerate(cls));
    pubs = cnt[:,1];

    adj = CSV.read("datasets/cora/cora.cites", header=0);
    hh = adj[:,1];
    tt = adj[:,2];
    @assert all(sort(pubs) .== sort(unique(union(hh,tt)))) "unexpected citation graph";
    pid2num = Dict(id=>num for (num,id) in enumerate(pubs));
    g = Graph(length(pid2num));
    for (h,t) in zip(hh,tt)
        add_edge!(g, pid2num[h], pid2num[t]);
    end

    y = map(x->cid2num[x], Array(cnt[:,end]));

    ff = Matrix(cnt[:,2:end-1]);
    f = [ff[i,:] for i in 1:size(ff,1)];

    if dim_reduction
        U,S,V = svds(hcat(f...); nsv=dim_embed)[1];
        UU = U .* sign.(sum(U,dims=1)[:])';
        f = [UU'*f_ for f_ in f];
        fbar = mean(f);
        f = [f_ - fbar for f_ in f];
    end

    return g, [adjacency_matrix(g)], y, f;
end

function read_fraud_detection(network_name)
    if network_name == "YelpChi"
        dat = matread("datasets/fraud_detection/YelpChi.mat")
        g0 = Graph(dat["net_rur"]);
    elseif network_name == "Amazon"
        dat = matread("datasets/fraud_detection/Amazon.mat")
        g0 = Graph(dat["net_upu"]);
    end

    lcc = sort(connected_components(g0), by=cc->length(cc))[end];
    g,_ = induced_subgraph(g0, lcc);

    ff = collect(dat["features"][lcc,:]);
    for i in 1:size(ff,2)
        ff[:,i] = std_normalize(ff[:,i]);
    end

    f = [ff[i,:] for i in 1:size(ff,1)];
    y = Int.(dat["label"][lcc] .+ 1.0);

    return g, [adjacency_matrix(g)], y, f;
end

function read_cropsim(type, year, num_predictors)
    meta = CSV.read("datasets/more-graphs/cropsim/cropsim-" * year * "-" * type * "-metadata.csv", header=0); n = size(meta,1);
    smat = CSV.read("datasets/more-graphs/cropsim/cropsim-" * year * "-" * type * ".smat", header=1);

    g = Graph(n);
    for i in 1:size(smat, 1)
        add_edge!(g, smat[i,1]+1, smat[i,2]+1);
    end

    y = std_normalize(log.(meta[!,1] .+ 1.0));

    ff = zeros(Float32, n,num_predictors);
    for i in 1:num_predictors
        ff[:,i] = std_normalize(log.(meta[!,i+1] .+ 1.0));
    end

    f = [ff[i,:] for i in 1:size(ff,1)];

    return g, [adjacency_matrix(g)], y, f;
end

function read_network(network_name)
    (p = match(r"ising_([0-9]+)_([0-9\.\-]+)_([0-9\.\-]+)$", network_name)) != nothing && return simulate_ising(parse(Int, p[1]), parse(Float64, p[2]), parse(Float64, p[3]));
    (p = match(r"county_(.+)_([0-9]+)$", network_name)) != nothing && return read_county(p[1], parse(Int, p[2]));
    (p = match(r"environment_(.+)_([0-9]+)$", network_name)) != nothing && return read_environment(p[1], parse(Int, p[2]));
    (p = match(r"ward_(.+)_([0-9]+)$", network_name)) != nothing && return read_ward(p[1], p[2]);
    (p = match(r"twitch_([0-9a-zA-Z]+)_([a-z]+)_([0-9]+)$", network_name)) != nothing && return read_twitch(p[1], parse(Bool, p[2]), parse(Int, p[3]));
    (p = match(r"sexual_([0-9]+)$", network_name)) != nothing && return read_sexual(parse(Int, p[1]));
    (p = match(r"Anaheim", network_name)) != nothing       && return read_transportation_network(network_name, 8, 1:2, [3,4,5,8], 6, [1,2,4], 1:416);
    (p = match(r"ChicagoSketch", network_name)) != nothing && return read_transportation_network(network_name, 7, 1:2, [3,4,5,8], 1, [1,2,3], 388:933);
    (p = match(r"Amazon", network_name)) != nothing && return read_fraud_detection(network_name);
    (p = match(r"cora_([a-z]+)_([0-9]+)", network_name)) != nothing && return read_cora(parse(Bool, p[1]), parse(Int, p[2]));
    (p = match(r"cropsim_([a-z]+)_([0-9]+)_([0-9]+)", network_name)) != nothing && return read_cropsim(p[1], p[2], parse(Int, p[3]));
end
