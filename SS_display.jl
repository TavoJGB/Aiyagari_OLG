##########################################################################
#### SUMMARY                                                          ####
##########################################################################

# Summary table
function SS_summary(conv::T, eco::Economia, her::Herramientas, sol::Solucion) where {T<:Real}
    @unpack β, α, δ, u′, a_min = eco
    @unpack n, mallaA, mallaZ, matSt, id = her
    @unpack a_pol, c_pol, r, w, A_agg, C_agg, K_agg, Y_agg, distr, resid = sol

    # Computing MPC
        # Auxiliary variables
        distr_2d = Array{Float64}(undef, n.a, Int64(n.N/n.a))
        cpol_2d = similar(distr_2d)
        distr_2d .= reshape(distr,n.a,:)
        cpol_2d .= reshape(c_pol,n.a,:)
        # First differences
        diff_cpol = cpol_2d[2:end,:] - cpol_2d[1:end-1,:]
        diff_apol = (mallaA[2:end] - mallaA[1:end-1])*conv
        # Compute average MPC
        mpc = sum(diff_cpol .* distr_2d[1:end-1, :] ./ diff_apol)

    return Dict{String,Float64}(
        "01. Annual interest rate" => r,
        "02. Gini" => Gini(a_pol, distr),
        "03. Aggregate consumption-to-GDP, C/Y" => (C_agg / Y_agg),
        "04. Capital-to-GDP, K/Y" => (K_agg / Y_agg),
        "05. Investment-to-GDP, I/Y" => (δ * K_agg / Y_agg),
        "06. Average MPC" => mpc,
        "07. Share of credit-constrained agents" => sum(distr_2d[1,:]))

end;

function print_dict(dict::Dict{String,Float64}; sep="",digits=4)::Nothing
    tuples = sort([(k,v) for (k,v) in dict], by = first)
    max_label_length = maximum(map(length∘first,tuples))
    for (k,v) in tuples
        k_ = lowercase(k)
        trail_space = repeat(' ',1+max_label_length-length(k))
        to_pct = (occursin("rate",k_)||occursin("share",k_)) 
        if to_pct
            println(k,sep*trail_space,round(100*v,digits=digits-2)," %")
        else
            println(k,sep*trail_space,round(v,digits=digits))
        end
    end
    return nothing
end;



##########################################################################
#### GRAPHS                                                           ####
##########################################################################

function generations_gridplot(nj::I, age1::I, xs::Vector{T}, ys::Vector{T},
                              her::Herramientas;
                              tit::String="",
                              xlab::String="", ylab::String="",
                              sumOne::Bool=false) where {T<:Real, I<:Integer}
    @unpack matSt, id, ind, n = her
    # Initialise relevant variables
    plots = Array{Plots.Plot}(undef, nj)
    # Create plots
    for jj=1:nj
        indJ = (matSt[:,id.j] .== jj)
        y_jj = [ys[indJ .& ind.Zmax] ys[indJ .& ind.Zmin]]
        if sumOne
            y_jj ./= sum(y_jj, dims=1)
        end
        plots[jj] = plot( xs, y_jj[:,1], label="high z")
                    plot!(xs, y_jj[:,2], label="low z")
                    xlabel!(xlab)
                    ylabel!(ylab)
                    title!(string("Age: ", age1 + (jj-1)*n.t, "-", age1 + jj*n.t - 1))
    end
    # Display them in tiled layout
    genplots = Plots.plot(
        # Global title: workaround to show global title (empty plot with annotation)
        Plots.scatter(ones(3), marker=0,markeralpha=0, annotations=(2, 1.0, Plots.text(tit)),axis=false, grid=false, leg=false,size=(200,100)),
        # Grid of policy functions
        Plots.plot(plots..., layout = nj),
        # Layout of title and policy functions
        layout=grid(2,1,heights=[0.1,0.9])
    )
    return genplots
end

function groups_gridplot(groupnames::Array{String}, groupvals::Array{Vector{I}}, varid::I,
    xs::Vector{T}, ys::Vector{T},
    her::Herramientas;
    tit::String="", xlab::String="", ylab::String="",
    sumOne::Bool=false) where {T<:Real, I<:Integer}

    @unpack matSt, id, ind, n = her

    # Initialise relevant variables
    nGr = length(groupnames)
    plots = Array{Plots.Plot}(undef, nGr)
    # Create plots
    for (iGr, grname, grvals) in zip(eachindex(groupnames), groupnames, groupvals)
        # Create indicator to nodes belonging to the group
        groupind = (matSt[:,varid] .∈ [grvals])
        # Prepare group data to be plotted
        y_group = [ys[groupind .& ind.Zmax] ys[groupind .& ind.Zmin]]
        if sumOne
            y_group ./= sum(y_group, dims=1)
        end
        # Create graph
        plots[iGr] =    plot( xs, y_group[:,1], label="high z")
                        plot!(xs, y_group[:,2], label="low z")
                        xlabel!(xlab)
                        ylabel!(ylab)
                        title!(grname)
    end
    # Display them in tiled layout
    groupplots = Plots.plot(
        # Global title: workaround to show global title (empty plot with annotation)
        Plots.scatter(ones(3), marker=0,markeralpha=0, annotations=(2, 1.0, Plots.text(tit)),axis=false, grid=false, leg=false,size=(200,100)),
        # Grid of policy functions
        Plots.plot(plots..., layout = nj),
        # Layout of title and policy functions
        layout=grid(2,1,heights=[0.1,0.9])
    )
    return groupplots
end

function group_distrplot(groupnames, groupinds, matSt, varid, xs, ys, func)
    # Initialise variables
    nvar = length(xs)
    group_ys = Matrix{Float64}(undef, nvar, 2)
    # Get data by group for each node in xs
    for ix in eachindex(xs)
        for (iGr, grind) in zip(eachindex(groupnames), groupinds)
            group_ys[ix,iGr] = func(ys[grind .& (matSt[:,varid].==ix)])
        end
    end
    group_distrplot = plot(xs, group_ys, label=groupnames)
    return group_distrplot
end


function SS_graphs(conv::T, eco::Economia, her::Herramientas,
    sol::Solucion, cfg::Configuracion)::Nothing where {T<:Real}
    @unpack age1, jRet = eco;
    @unpack n, matSt, mallaA, mallaζ, mallaZ, id, ind = her;
    @unpack r, w, c_pol, a_pol, value, inc_l, inc_a, distr = sol;
    @unpack doubBin, plotsiz, gridplotsiz = cfg;

# FOLDER
    # create directory for images, if it does not exist yet
    mkpath("./Figuras");

# AUXILIARY VARIABLES
    if doubBin # double-sized bins
        ageBins = Int64.(age1 .+ 2*n.t*((1:round(n.j/2)).-1))                   # Correspondence age-period
        stJ     = ceil.(matSt[:,id.j]/2)                                        # Indexes for vector accumulation
        ageBinLabs = string.(ageBins) .* "-" .* string.(ageBins .+ 2*n.t .- 1)  # Labels for bar plots
    else
        ageBins = Int64.(age1 .+ n.t*((1:n.j).-1))                              # Correspondence age-period
        stJ     = matSt[:,id.j]                                                 # Indexes for vector accumulation
        ageBinLabs = string.(ageBins) .* "-" .* string.(ageBins .+ n.t .- 1)    # Labels for bar plots
    end
    masaJ   = vcat([sum(distr[jj .∈ stJ,:], dims=1) for jj in unique(stJ)]...)


# CONSUMPTION I - policy functions
    # Main
    generations_gridplot(jRet-1, age1, conv/1000*mallaA, c_pol/1000, her, tit="Policy functions: consumption (×10³ €)", xlab="asset holdings (×10³ €)", ylab="consumption (×10³ €)")
    # Other settings
    plot!(size=gridplotsiz)
    # Save graph
    Plots.savefig("./Figuras/SS_c_pol.png")


# CONSUMPTION II - life cycle
    # Main
    bar(ageBinLabs, vcat([sum((c_pol/1000 .* distr)[jj .∈ stJ,:], dims=1) for jj in unique(stJ)]...) ./ masaJ, label="")
    # Titles and labels
    xlabel!("Age group")
    ylabel!("Average consumption (×10³ €)")
    title!("Life cycle: consumption (×10³ €)")
    # Other settings
    plot!(size=plotsiz)
    # Save graph
    Plots.savefig("./Figuras/SS_c_life.png")


# SAVINGS I - policy functions
    # Main
    generations_gridplot(jRet-1, age1, conv/1000*mallaA, a_pol/1000, her; tit="Policy functions: savings (×10³ €)", xlab="asset holdings (×10³ €)", ylab="savings (×10³ €)")
    # Other settings
    plot!(size=gridplotsiz)
    # Save graph
    Plots.savefig("./Figuras/SS_a_pol.png")


# SAVINGS II - life cycle
    # Main
    bar(ageBinLabs, vcat([sum((a_pol/1000 .* distr)[jj .∈ stJ,:], dims=1) for jj in unique(stJ)]...) ./ masaJ, label="")
    hline!([0.0], label="")
    # Titles and labels
    xlabel!("Age group")
    ylabel!("Average savings (×10³ €)")
    title!("Life cycle: savings (×10³ €)")
    # Other settings
    plot!(size=plotsiz)
    # Save graph
    Plots.savefig("./Figuras/SS_a_life.png")


# ASSET DISTRIBUTION I - by productivity group
    # Main
    generations_gridplot(jRet-1, age1, conv/1000*mallaA, distr, her; tit="Wealth distribution\nby age group", xlab="asset holdings (×10³ €)", sumOne=true)
    # Other settings
    plot!(size=gridplotsiz)
    # Save graph
    savefig("./Figuras/SS_a_dist_z.png")


# ASSET DISTRIBUTION II - by life stage
    # Main
    group_distrplot(["Workers" "Retirees"], [ind.wrk, ind.ret], matSt, id.a, conv/1000*mallaA, distr, sum)
    # Title and labels
    title!("Wealth distribution\nby life stage")
    xlabel!("asset holdings (×10³ €)")
    # Other settings
    plot!(size=plotsiz)
    # Save graph
    savefig("./Figuras/SS_a_dist_lc.png")    


# INCOME - by life stage
    # Accumulate by groups (1st column: capital income. 2nd column: labour income)
    jInc = [vcat([sum((inc_a.*distr)[jj .∈ stJ,:], dims=1) for jj in unique(stJ)]...) vcat([sum((inc_l.*distr)[jj .∈ stJ,:], dims=1) for jj in unique(stJ)]...)]
    # Bar plot
    groupedbar(ageBinLabs, jInc ./ (1000*masaJ), bar_position = :stack, label=["Capital income" "Labour income"])
    hline!([0.0], label="")
    # Titles and labels
    xlabel!("Age group")
    ylabel!("Average income (×10³ €)")
    title!("Life cycle: income (×10³ €)")
    # Other settings
    plot!(size=plotsiz)
    # Save graph
    Plots.savefig("./Figuras/SS_inc_life.png")


# VALUE - by age groups
    # Main
    generations_gridplot(jRet-1, age1, conv/1000*mallaA, value, her; tit="Value functions", xlab="asset holdings (×10³ €)")
    # Other settings
    plot!(size=gridplotsiz)
    # Save graph
    Plots.savefig("./Figuras/SS_value.png")


    return nothing
end



##########################################################################
#### TABLES                                                           ####
##########################################################################

function SS_tables(her::Herramientas, sol::Solucion; nquants::I=5, top::I=10)  where {I<:Integer}
    @unpack mallaA, matSt, id, ind = her
    @unpack inc_l, distr = sol

    # Cross section: labour income and wealth quantiles
        # compute quantiles
        inc_l_quants, inc_l_top = get_quants(nquants, inc_l[ind.wrk], distr[ind.wrk], top)
        a_quants, a_top = get_quants(nquants, mallaA[matSt[:,id.a]], distr, top)
        # display tables
        println("\nDISPERSION (cross-section)")
        println("Labour income")
        println("-\tShare of total labour income by labour income quintile:\n\t" * join(string.(fmt.(100*inc_l_quants',1)) .* " %\t"))
        println("-\tShare of total labour income received by top $top%:\n\t" * string.(fmt.(100*inc_l_top,1)) * " %")
        println("Wealth")
        println("-\tShare of total wealth by wealth quintile:\n\t" * join(string.(fmt.(100*a_quants',1)) .* " %\t"))
        println("-\tShare of total wealth held by top $top%:\n\t" * string.(fmt.(100*a_top,1)) * " %")
end