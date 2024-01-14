##########################################################################
#### INTERPOLATION                                                    ####
##########################################################################

# Scalar
# Vector
function getWeights(x::T, y::Vector{T}; metodo="extrap") where {T<:Real}

    # The function returns three vectors:
    # - lower: position in y of the nearest element below x.
    # - upper: position in y of the nearest element above x.
    # - weight: weight of lower in the linear combination that gives x as a function of y[lower] and y[upper].

    # Finding elements in y immediately above and below x
        # Number of elements in y:
            sizY = size(y,1)
        # Find lower neighbour in y
            lower = searchsortedlast(y, x)
            #lower is the largest index such that y[lower]≤x (and hence y[lower+1]>x). Returns 0 if x≤y[1]. y sorted.
        # Elements beyond the boundaries of y
            lower = clamp(lower, 1, sizY-1)
        # Corresponding upper neighbour
            upper = lower+1
    # Computing the weight of the element below
        weight = (y[upper] - x) / (y[upper] - y[lower])
        # the weight for the upper element is (1 - weight)

    # Cutting values
    if metodo=="cap"
        weight = clamp(weight, 0, 1)
    end

    # returns interpolated value and corresponding index
    return lower, upper, weight
end;

# Vector
function getWeights(x::Vector{T}, y::Vector{T}; metodo="extrap") where {T<:Real}

    # The function returns three vectors:
    # - lower: position in y of the nearest element below each x.
    # - upper: position in y of the nearest element above each x.
    # - weight: weight of lower in the linear combination that gives x.

    # Finding elements in y immediately above and below x
        # Number of elements in each vector:
            sizX = size(x,1)
            sizY = size(y,1)
        # Initialise vectors
            lower = Array{Int64}(undef, sizX)
            upper = similar(lower)
            weight = similar(lower)
        # Find lower elements for each of them
            lower = clamp.(searchsortedlast.(Ref(y), x), 1, sizY-1)
            # lower is the largest index such that y[lower]≤x (and hence y[lower+1]>x). Returns 1 if x≤y[1] and sizY-1 if x≥y[end-1].
            # y must be sorted.
        # Corresponding upper neighbour
            upper = lower.+1
    # Computing the weight of the element below
        weight = (y[upper] .- x) ./ (y[upper] .- y[lower])
        # the weight for the upper element is (1 - weight)

    # Cutting values
    if metodo=="cap"
        weight = clamp.(weight, 0, 1)
    end

    # returns interpolated value and corresponding index
    return lower, upper, weight
end;

# Interpolation
function interpLinear(x::T,
    y::Vector{T},
    z::Vector{T};
    metodo::String="extrap")::T  where {T<:Real}

    lower, upper, weight = getWeights(x,y;metodo=metodo)

    # returns interpolated value and corresponding index
    return weight*z[lower] + (1 - weight)*z[upper]
end

# Vector version
function interpLinear(x::Vector{T},
    y::Vector{T},
    z::Vector{T};
    metodo::String="extrap")::Vector{T}  where {T<:Real}

    lower, upper, weight = getWeights(x,y;metodo=metodo)

    # returns interpolated value and corresponding index
    return weight.*z[lower] .+ (1 .- weight).*z[upper]
end



##########################################################################
#### GINI                                                             ####
##########################################################################

# Gini
function Gini(ys::Vector{T}, pys::Vector{T})::T where{T<:Real}
    @assert size(ys)==size(pys)
    iys = sortperm(ys)

    ys_Gini = similar(ys)
    ys_Gini .= ys[iys]
    pys_Gini = similar(pys)
    pys_Gini .= pys[iys]
    Ss = [zero(T); cumsum(ys_Gini.*pys_Gini)]
    return one(T) - sum(pys_Gini.*(Ss[1:end-1].+Ss[2:end]))/Ss[end]
end
function Gini(ys::Matrix{T}, pys::Matrix{T})::T where{T<:Real}
    @assert size(ys)==size(pys)
    return Gini(ys[:],pys[:])
end;



##########################################################################
#### FORMAT                                                           ####
##########################################################################

# Formatting function
function fmt(x::I, prec::J=2) where{I<:Integer,J<:Integer}
    return x
end
function fmt(x::T, prec::J=2) where{T<:Real,J<:Integer}
    return round(x,digits=prec)
end;



##########################################################################
#### QUANTILES                                                        ####
##########################################################################

function get_quants(nq::I, data::Vector{T}, distr::Vector{T}, top::I) where {T<:Real, I<:Integer}
    # Preparation: position of divisions
    divs = range(0,1;length=nq+1)[2:end]
    # Rank nodes from lower to higher values
    iSort = sortperm(data)
    sorted = data[iSort]
    # Cumulative distribution
    sorted_distr = distr[iSort]
    cumdistr = cumsum(sorted_distr) / sum(sorted_distr)
    # Shares associated to the conditional distribution
    cumSh = cumsum(sorted .* sorted_distr / sum(sorted .* sorted_distr))
    # Find last element with positive distribution
    iLast = findlast(sorted_distr .> 1e-8)
    # Interpolate on the quantiles of interest
    quants = interpLinear(collect(divs), cumdistr[1:iLast], cumSh[1:iLast]; metodo="cap")
    quants[2:end] .-= quants[1:(end-1)]
    # Interpolate share of the top
    sharetop = 1-interpLinear(1-top/100, cumdistr[1:iLast], cumSh[1:iLast]; metodo="cap")

    return quants, sharetop
end



##########################################################################
#### QUANTILES                                                        ####
##########################################################################

function tiled_graph(plots::Vector{Plots.Plot}; tit::String="")
    # Auxiliary: number of plots
    np = size(plots,1)
    # Display them in tiled layout
    tiledplot = Plots.plot(
        # Global title: workaround to show global title (empty plot with annotation)
        Plots.scatter(ones(3), marker=0,markeralpha=0, annotations=(2, 1.0, Plots.text(tit)),axis=false, grid=false, leg=false,size=(200,100)),
        # Grid of policy functions
        Plots.plot(plots..., layout = np),
        # Layout of title and policy functions
        layout=grid(2,1,heights=[0.1,0.9])
    )
    return tiledplot
end