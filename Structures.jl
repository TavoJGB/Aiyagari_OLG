##########################################################################
#### MODEL PARAMETERS                                                 ####
##########################################################################

struct Economia{Tr<:Real, Ti<:Integer}
    # Parameters: model
        # Age structure
        nt::Ti               # Years per period
        jGap::Ti             # Period gap between generations
        jRet::Ti             # Retirement period
        age1::Ti             # Age in first period
        # Household
        γ::Tr                # CRRA
        β::Tr                # Discount factor
        ρz::Tr               # Persistence of the idiosyncratic shock
        σz::Tr               # Variance of the idiosyncratic shock
        # Producer
        α::Tr                # Share of capital
        δ::Tr                # Depreciation rate
        # Markets
        a_min::Tr            # Borrowing constraint
    
    # Functions: Utility
        u::Function
        u′::Function
        inv_u′::Function
    end
    
    # CONSTRUCTOR
    function Economia(a_min::Tr, nt::Ti, jGap::Ti, jRet::Ti, age1::Ti,
        γ::Tr, β::Tr, ρz::Tr, σz::Tr, α::Tr, δ::Tr) where {Tr<:Real, Ti<:Integer}
    
        # Utility functions
        u  = c::Tr -> (γ ≈ 1.0) ? log.(c) : (c .^ (one(Tr) - γ) - one(Tr)) ./ (one(Tr) - γ)::Tr
        u′ = c::Tr -> c .^ (-γ)::Tr
        inv_u′ = up::Tr -> up .^ (-one(Tr) / γ)::Tr
    
        # Return the Economy using the natural constructor 
        return Economia(nt, jGap, jRet, age1, γ, β^nt, ρz^nt, σz, α, 1-(1-δ)^nt, a_min, u, u′, inv_u′)
    end;
    
    
    
    ##########################################################################
    #### TOOLS                                                            ####
    ##########################################################################
    
    struct Herramientas{Tr<:Real, Ti<:Integer}
    # Parameters: numerical solution
        # Dimensions
        n::NamedTuple{(:t,:j,:z,:a,:st,:N,:red,:aj), Tuple{Int64,Int64,Int64,Int64,Int64,Int64,Int64,Vector{Ti}}}
        # Capital (log)grid
        a_max::Tr                                        # Maximum asset holdings
        mallaA::Vector{Tr}                               # Assets grid
        # Productivity grid
        mallaZ::Vector{Tr}                               # Productivity states
        Πz::Matrix{Tr}                                   # Transition matrix
        Sz::Vector{Tr}                                   # Productivity long-run distribution
        # Life-cycle
        mallaζ::Vector{Tr}                               # Life-cycle productivity
        # States
        id::NamedTuple{(:j,:z,:a,:a0), NTuple{4,Ti}}     # Indexes
        matSt::Matrix{Ti}                                # Matrix of states
        # Indicators of state
        ind::NamedTuple{(:newby,:newpt,:wrk,:ret,:last,:Zmax,:Zmin), NTuple{7,BitVector}}
        # Auxiliary (reduced) state pace <- ignoring decision-state variables
        redQ::SparseMatrixCSC{Tr,Ti}                      # Reduced transition matrix
        redSt::Matrix{Ti}                                # Reduced matrix of states
    end;
    
    # Constructor
    function Herramientas(na_min::Ti, na_max::Ti, # Nodes in asset grid (exact number depends on age)
                          nz::Ti,                # Nodes in productivity grid
                          a_max::Tr,             # Upper bound in assets grid (for generation saving most)
                          nj::Ti,                # Number of generations (and last period alive)
                          curvA::Tr,             # Curvature of the asset grid
                          eco::Economia         # Model parameters
                         ) where {Tr<:Real, Ti<:Integer}
    
    # Unpack relevant variables
        @unpack nt,age1,jGap,jRet,a_min,ρz,σz = eco
    
    # States
        # Number of states in the model
        nst = 3 # age, productivity, assets
        # Asset states: depend on age
        naj = round.(Int64,range(na_max, na_min; length=(jRet-1))[abs.((1:nj) .- (jRet-1)) .+ 1])
        # Total states (n.N)
        nN  = ( nz*sum(naj[1:(jRet-1)]) +   # Working-age generations
                sum(naj[jRet:end]) )        # Retired agents
    
    # Asset (log)grid
        aux    = range(0, stop=log(a_max+1-a_min), length=na_max)
        mallaA  = @. exp(aux) - (1-a_min)
        mallaA = a_min .+ (a_max - a_min) * (range(0.0, 1.0, length=na_max)) .^ curvA
        if 0 ∉ mallaA # Ensure that 0 belongs in the grid
            mallaA[findfirst(mallaA.>0)] = 0
        end
        # Consequence: minimum assets can't be >0
        @assert a_min <= 0
        
    # Productivity process
        mc = rouwenhorst(nz, ρz, σz)
        Trans = collect(mc.p')
        Trans[findall(x -> x <= 5 * 10^-5, Trans)] .= zero(Trans[1, 1])
        for i = 1:nz
            Trans[i, i] += one(Trans[1, 1]) - sum(Trans, dims=1)[i]
        end
        Sz = (Trans^100000)[:, 1]
        # Normalización
        endow = exp.(mc.state_values)
        Πz = collect(Trans')
        mallaZ = endow ./ dot(Sz, endow) # ensuring L=1
    
    # Life-cycle productivity
        # Parameters (from second order polynomial regression)
        ζ1 = -118816.67180;
        ζ2 = 7367.34773;
        ζ3 = -76.85439;
        # Initialise variable
        mallaζ = Array{Float64}(undef, nj)
        # Working ages
        auxAge = (age1 + nt/2 .+ nt*(1:(jRet-1)));  # Add nt/2 years to be in the middle of the bin
        # Life-cycle profile of earnings
        mallaζ[1:(jRet-1)] = ζ1 .+ ζ2*auxAge + ζ3*auxAge.^2
        mallaζ[jRet:nj] .= 0                        # No production once retired
        # Average productivity while working: normalisation to one
        mallaζ .= mallaζ / mean(mallaζ[1:(jRet-1)]);
    
    # Indexes
        id  = (j=1, z=2, a=3,
               a0=(i = findfirst(mallaA.==0); isnothing(i) ? 0 : i))
        @assert id.a0 < na_min
        # 1 = age
        # 2 = idiosyncratic productivity
        # 3 = asset holdings
    
    # Matrix of states
        # Basis
        matSt           = zeros(Ti, nj*nz*na_max, nst)
        matSt[:,id.j]   = kron( 1:nj, ones(nz*na_max) )
        matSt[:,id.z]   = kron( ones(nj), kron( 1:nz, ones(na_max) ) )
        matSt[:,id.a]   = kron( ones(nz*nj), 1:na_max )
        # Corrections
            # matSt = matSt[(matSt[:,id.j] .> 1) .| (matSt[:,id.a] .== id.a0),:]   # Newcomers start with no assets
            matSt = matSt[(matSt[:,id.j] .< jRet) .| (matSt[:,id.z] .== 1), :]   # Productivity doesn't matter after retirement
            matSt = vcat([matSt[(matSt[:,id.a] .<= naj[jj]) .& (matSt[:,id.j] .== jj), :] for jj=1:nj]...)  # Reduce asset grids for generations saving less
        # Verifications
        @assert nN==size(matSt,1)
    
    # Indicators
        ind = ( # Age structure
                newby   = matSt[:,id.j].==1, 
                newpt   = matSt[:,id.j].==jGap,
                wrk     = matSt[:,id.j].<jRet,
                ret     = matSt[:,id.j].>=jRet,
                last    = matSt[:,id.j].==nj,
                # Productivity
                Zmax    = matSt[:,id.z] .== nz,
                Zmin    = matSt[:,id.z] .== 1
              )
    
    # State transitions (reduced Q) - abstracting from decision state variables
        # Next period states (ignoring current assets, both own and parents')
        redSt = matSt[matSt[:,id.a].==id.a0,:]  # They indicate the state in each column of Qred
        nred  = size(redSt,1);
        jNext = redSt[:,id.j]';                 # Age at each position of the transition matrix
        zNext = redSt[:,id.z]';                 # z' at each position of the transition matrix
        # Probability of transition to next period's stochastic states
            # Initialise matrix
            redQ = spzeros(Float64,nN,nred)
            # Age
            redQ .= ((matSt[:,id.j].+1).==jNext)
            # Productivity
            auxTr_Z = spzeros(Float64,nN,nred)
            for col=1:nred
                auxTr_Z[matSt[:,id.j].<(jRet-1),col] .= Πz[matSt[matSt[:,id.j].<(jRet-1),id.z], zNext[col]]
                # Correction: productivity does not change since retirement
                auxTr_Z[matSt[:,id.j].>=(jRet-1),col] .= 1
            end
            redQ = redQ .* auxTr_Z
            # New generation enters when parents leave period jGap (ind.newpt)
            redQ[ind.newpt, redSt[:,id.j].==1] .+= Sz'
            # Verification
            @assert all(sum(redQ,dims=2) .≈ 1.0 .+ ind.newpt - ind.last)
        
    # Tuple of dimensions
        # Named tuple with dimensions
        n   = (t=nt, j=nj, z=nz, a=na_max, st=nst, N=nN, red=nred, aj=naj)
    
    # Return Herramientas
        return Herramientas(n, a_max, mallaA, mallaZ, Πz, Sz, mallaζ, id, matSt, ind, redQ, redSt)
    end;
    
    
    
    ##########################################################################
    #### MODEL CONSTRUCTOR                                                ####
    ##########################################################################
    
    # Function nesting the previous two structures
    function Model(
        na_min::Ti, na_max::Ti,         # Nodes in asset grid (exact number depends on age)
        nz::Ti,                         # Nodes in productivity grid
        a_min::Tr, a_max::Tr;           # Borrowing constraint and upper bound in assets grid
        # Age structure
        nj      = 16,                   # Number of generations (and last period alive)
        nt      = 5,                    # Years per period
        jGap    = 7,                    # Period gap between generations
        jRet    = 10,                   # Retirement period
        age1    = 20,                   # Age in first period
        # Household
        γ       = 1.5,                  # CRRA
        β       = 0.94,                 # Discount factor
        ρz      = 0.966,                # Persistence of the idiosyncratic shock
        σz      = 0.15,                 # Variance of the idiosyncratic shock
        # Producer
        α       = 1/3,                  # Share of capital
        δ       = 0.1,                  # Depreciation rate
        # Numerical solution
        curvA   = 4.0                   # Curvature of the asset grid
    ) where {Tr<:Real, Ti<:Integer}
    
        # Economy (model parameters)
        eco = Economia(a_min, nt, jGap, jRet, age1, γ, β, ρz, σz, α, δ)
        # Tools (grids, probabilities, etc.)
        her = Herramientas(na_min, na_max, nz, a_max, nj, curvA, eco)
        
        # Construct the structure
        return eco, her
    end;
    
    
    
    ##########################################################################
    #### CONFIGURATION                                                    ####
    ##########################################################################
    
    # tolerance and other tools for the computation of the numerical solution
    struct Configuracion{Tr<:Real, Ti<:Integer}
        # General equilibrium: relaxation, tolerance and maximum iterations
        rlx_SGE::Tr              # weight of initial guess in relaxation algorithm
        tol_SGE::Tr
        maxit_SGE::Ti
            # # EGM: tolerance and maximum iterations
            # tol_EGM::Tr
            # maxit_EGM::Ti
        # Golden search
        tol_GS::Tr               # tolerance
        # Stationary distribution: tolerance and maximum iterations
        tol_SSdis::Tr
        maxit_SSdis::Ti
        # Final verifications
        tol_check::Tr
        # Other parameters for the numerical solution
        c_min::Tr                # minimum consumption allowed
        penal::Tr                # penalty to ensure minimum consumption
        # Graphs
        doubBin::Bool           # groups of two generations
        plotsiz::Vector{Ti}      # size for normal plots
        gridplotsiz::Vector{Ti}  # size for grids with subplots
    end;
    
    # Constructor
    function Configuracion(rlx_SGE::Tr;
        tol_SGE::Tr=1e-6, maxit_SGE::Ti=100,
        #tol_EGM::Tr=1e-10, maxit_EGM::Ti=100000,
        tol_GS::Tr=1e-10,
        tol_SSdis::Tr=1e-10, maxit_SSdis::Ti=100000,
        tol_check::Tr=1e-5,
        c_min::Tr=1e-2, penal::Tr=-1e6,
        doubBin::Bool=false,
        plotsiz::Vector{Ti}=[800,500], gridplotsiz::Vector{Ti}=[1200,900]
    ) where {Tr<:Real,Ti<:Integer}
    
    # Construct the structure
    return Configuracion{Tr,Ti}(rlx_SGE, tol_SGE, maxit_SGE, tol_GS, tol_SSdis, maxit_SSdis,
                              tol_check, c_min, penal,
                              doubBin, plotsiz, gridplotsiz)
    end;
    
    
    ##########################################################################
    #### SOLUTION                                                         ####
    ##########################################################################
    
    mutable struct Solucion{Tr<:Real, Ti<:Integer}
        r::Tr                            # interest rate
        w::Tr                            # wage rate
        a_pol::Vector{Tr}                # policy function for savings on the state space
        c_pol::Vector{Tr}                # policy function for consumption on the state space
        value::Vector{Tr}                # value function on the state space
        inc_l::Vector{Tr}                # households' labour income
        inc_a::Vector{Tr}                # households' capital income
        A_agg::Tr                        # aggregate savings
        C_agg::Tr                        # aggregate consumption
        K_agg::Tr                        # aggregate capital
        L_agg::Tr                        # aggregate labor supply (in efficient units)
        Y_agg::Tr                        # GDP
        Q_mat::SparseMatrixCSC{Tr,Ti}     # Q-transition matrix
        distr::Vector{Tr}                # stationary distribution over the states
        resid::Vector{Tr}                # Euler equation residuals
    end;
    
    # Constructor
    function Solucion(r::Tr, w::Tr, eco::Economia, her::Herramientas, a_pol::Vector{Tr},
                      c_pol::Vector{Tr}, value::Vector{Tr}, Q_mat::SparseMatrixCSC{Tr,Ti}, distr::Vector{Tr}
                      ) where {Tr<:Real, Ti<:Integer}
        @unpack α, β, δ, u′ = eco
        @unpack n, mallaA, mallaZ, mallaζ, matSt, id = her
    
        # Household
            # Income variables
            inc_l = w*mallaζ[matSt[:,id.j]].*mallaZ[matSt[:,id.z]]                  # labour income
            inc_a = r*mallaA[matSt[:,id.a]]                                         # capital income
            # Aggregates
            A_agg = sum(distr .* a_pol)                                             # savings
            C_agg = sum(distr .* c_pol)                                             # consumption
            L_agg = sum(distr .* mallaζ[matSt[:,id.j]] .* mallaZ[matSt[:,id.z]])    # effective labour
    
        # Producer
        K_agg = L_agg * ((r + δ) / α)^(1.0 / (α - 1.0)) #aggregate capital
        Y_agg = K_agg^α * L_agg^(1-α)
    
        # Verification: Euler equations
        resid = u′.(c_pol) - β * (1+r) * Q_mat' * u′.(c_pol)
    
        # Construct the structure
        return Solucion(r, w, a_pol, c_pol, value, inc_l, inc_a, A_agg, C_agg, K_agg, L_agg, Y_agg, Q_mat, distr, resid)
    end;