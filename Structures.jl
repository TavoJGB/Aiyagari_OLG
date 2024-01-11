##########################################################################
#### MODEL PARAMETERS                                                 ####
##########################################################################

struct Economia{T<:Real, I<:Integer}
    # Parameters: model
        # Age structure
        nt::I               # Years per period
        jGap::I             # Period gap between generations
        jRet::I             # Retirement period
        age1::I             # Age in first period
        # Household
        γ::T                # CRRA
        β::T                # Discount factor
        ρz::T               # Persistence of the idiosyncratic shock
        σz::T               # Variance of the idiosyncratic shock
        # Producer
        α::T                # Share of capital
        δ::T                # Depreciation rate
        # Markets
        a_min::T            # Borrowing constraint
    
    # Functions: Utility
        u::Function
        u′::Function
        inv_u′::Function
    end
    
    # CONSTRUCTOR
    function Economia(a_min::T, nt::I, jGap::I, jRet::I, age1::I,
        γ::T, β::T, ρz::T, σz::T, α::T, δ::T) where {T<:Real, I<:Integer}
    
        # Utility functions
        u  = c::T -> (γ ≈ 1.0) ? log.(c) : (c .^ (one(T) - γ) - one(T)) ./ (one(T) - γ)::T
        u′ = c::T -> c .^ (-γ)::T
        inv_u′ = up::T -> up .^ (-one(T) / γ)::T
    
        # Return the Economy using the natural constructor 
        return Economia(nt, jGap, jRet, age1, γ, β^nt, ρz^nt, σz, α, 1-(1-δ)^nt, a_min, u, u′, inv_u′)
    end;
    
    
    
    ##########################################################################
    #### TOOLS                                                            ####
    ##########################################################################
    
    struct Herramientas{T<:Real, I<:Integer}
    # Parameters: numerical solution
        # Dimensions
        n::NamedTuple{(:t,:j,:z,:a,:st,:N,:red), NTuple{7,I}}
        # Capital (log)grid
        a_max::T                                        # Maximum asset holdings
        mallaA::Vector{T}                               # Assets grid
        # Productivity grid
        mallaZ::Vector{T}                               # Productivity states
        Πz::Matrix{T}                                   # Transition matrix
        Sz::Vector{T}                                   # Productivity long-run distribution
        # Life-cycle
        mallaζ::Vector{T}                               # Life-cycle productivity
        # States
        id::NamedTuple{(:j,:z,:a,:a0), NTuple{4,I}}     # Indexes
        matSt::Matrix{I}                                # Matrix of states
        # Indicators of state
        ind::NamedTuple{(:newby,:newpt,:wrk,:ret,:last,:Zmax,:Zmin), NTuple{7,BitVector}}
        # Auxiliary (reduced) state pace <- ignoring decision-state variables
        redQ::SparseMatrixCSC{T,I}                      # Reduced transition matrix
        redSt::Matrix{I}                                # Reduced matrix of states
    end;
    
    # Constructor
    function Herramientas(na::I, nz::I, # Nodes in capital and productivity grids
                          a_max::T,     # Borrowing constraint and upper bound in assets grid
                          nj::I,        # Number of generations (and last period alive)
                          curvA::T,     # Curvature of the asset grid
                          eco::Economia # Model parameters
                         ) where {T<:Real, I<:Integer}
    
    # Unpack relevant variables
        @unpack nt,age1,jGap,jRet,a_min,ρz,σz = eco
    
    # States
        # Number of states in the model
        nst = 3 # age, productivity, assets
        # Total states (n.N)
        nN  = ( (jRet-1)*nz*na +    # Working-age generations
                (nj-jRet+1)*na )    # Retired agents
    
    # Asset (log)grid
        aux    = range(0, stop=log(a_max+1-a_min), length=na)
        mallaA  = @. exp(aux) - (1-a_min)
        mallaA = a_min .+ (a_max - a_min) * (range(0.0, 1.0, length=na)) .^ curvA
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
        # 1 = age
        # 2 = idiosyncratic productivity
        # 3 = asset holdings
    
    # Matrix of states
        # Basis
        matSt           = zeros(I, nj*nz*na, nst)
        matSt[:,id.j]   = kron( 1:nj, ones(nz*na) )
        matSt[:,id.z]   = kron( ones(nj), kron( 1:nz, ones(na) ) )
        matSt[:,id.a]   = kron( ones(nz*nj), 1:na )
        # Corrections
        # matSt = matSt[(matSt[:,id.j] .> 1) .| (matSt[:,id.a] .== id.a0),:]   # Newcomers start with no assets
        matSt = matSt[(matSt[:,id.j] .< jRet) .| (matSt[:,id.z] .== 1), :]   # Productivity doesn't matter after retirement
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
            redQ .= ((matSt[:,id.j].+1).==jNext);
            # Productivity
            auxTr_Z = spzeros(Float64,nN,nred);
            for col=1:nred
                auxTr_Z[matSt[:,id.j].<(jRet-1),col] .= Πz[matSt[matSt[:,id.j].<(jRet-1),id.z], zNext[col]];
                # Correction: productivity does not change since retirement
                auxTr_Z[matSt[:,id.j].>=(jRet-1),col] .= 1;
            end
            redQ = redQ .* auxTr_Z;
            # New generation enters when parents leave period jGap (ind.newpt)
            redQ[ind.newpt, redSt[:,id.j].==1] .+= Sz';
            # Verification
            @assert all(sum(redQ,dims=2) .≈ 1.0 .+ ind.newpt - ind.last)
        
    # Tuple of dimensions
        # Named tuple with dimensions
        n   = (t=nt, j=nj, z=nz, a=na, st=nst, N=nN, red=nred)
    
    # Return Herramientas
        return Herramientas(n, a_max, mallaA, mallaZ, Πz, Sz, mallaζ, id, matSt, ind, redQ, redSt)
    end;
    
    
    
    ##########################################################################
    #### MODEL CONSTRUCTOR                                                ####
    ##########################################################################
    
    # Function nesting the previous two structures
    function Model(
        na::I, nz::I,                   # Nodes in capital and productivity grids
        a_min::T, a_max::T;             # Borrowing constraint and upper bound in assets grid
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
    ) where {T<:Real, I<:Integer}
    
    # Economy (model parameters)
    eco = Economia(a_min, nt, jGap, jRet, age1, γ, β, ρz, σz, α, δ)
    # Tools (grids, probabilities, etc.)
    her = Herramientas(na, nz, a_max, nj, curvA, eco)
    
    # Construct the structure
    return eco, her
    end;
    
    
    
    ##########################################################################
    #### CONFIGURATION                                                    ####
    ##########################################################################
    
    # tolerance and other tools for the computation of the numerical solution
    struct Configuracion{T<:Real, I<:Integer}
        # General equilibrium: relaxation, tolerance and maximum iterations
        rlx_SGE::T              # weight of initial guess in relaxation algorithm
        tol_SGE::T
        maxit_SGE::I
            # # EGM: tolerance and maximum iterations
            # tol_EGM::T
            # maxit_EGM::I
        # Golden search
        tol_GS::T               # tolerance
        # Stationary distribution: tolerance and maximum iterations
        tol_SSdis::T
        maxit_SSdis::I
        # Final verifications
        tol_check::T
        # Other parameters for the numerical solution
        c_min::T                # minimum consumption allowed
        penal::T                # penalty to ensure minimum consumption
        # Graphs
        doubBin::Bool           # groups of two generations
        plotsiz::Vector{I}      # size for normal plots
        gridplotsiz::Vector{I}  # size for grids with subplots
    end;
    
    # Constructor
    function Configuracion(rlx_SGE::T;
        tol_SGE::T=1e-6, maxit_SGE::I=100,
        #tol_EGM::T=1e-10, maxit_EGM::I=100000,
        tol_GS::T=1e-10,
        tol_SSdis::T=1e-10, maxit_SSdis::I=100000,
        tol_check::T=1e-5,
        c_min::T=1e-2, penal::T=-1e6,
        doubBin::Bool=false,
        plotsiz::Vector{I}=[800,500], gridplotsiz::Vector{I}=[1200,900]
    ) where {T<:Real,I<:Integer}
    
    # Construct the structure
    return Configuracion{T,I}(rlx_SGE, tol_SGE, maxit_SGE, tol_GS, tol_SSdis, maxit_SSdis,
                              tol_check, c_min, penal,
                              doubBin, plotsiz, gridplotsiz)
    end;
    
    
    ##########################################################################
    #### SOLUTION                                                         ####
    ##########################################################################
    
    mutable struct Solucion{T<:Real, I<:Integer}
        r::T                            # interest rate
        w::T                            # wage rate
        a_pol::Vector{T}                # policy function for savings on the state space
        c_pol::Vector{T}                # policy function for consumption on the state space
        value::Vector{T}                # value function on the state space
        inc_l::Vector{T}                # households' labour income
        inc_a::Vector{T}                # households' capital income
        A_agg::T                        # aggregate savings
        C_agg::T                        # aggregate consumption
        K_agg::T                        # aggregate capital
        L_agg::T                        # aggregate labor supply (in efficient units)
        Y_agg::T                        # GDP
        Q_mat::SparseMatrixCSC{T,I}     # Q-transition matrix
        distr::Vector{T}                # stationary distribution over the states
        resid::Vector{T}                # Euler equation residuals
    end;
    
    # Constructor
    function Solucion(r::T, w::T, eco::Economia, her::Herramientas, a_pol::Vector{T},
                      c_pol::Vector{T}, value::Vector{T}, Q_mat::SparseMatrixCSC{T,I}, distr::Vector{T}
                      ) where {T<:Real, I<:Integer}
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