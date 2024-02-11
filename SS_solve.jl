##########################################################################
#### GOLDEN SEARCH                                                    ####
##########################################################################

function golden_policies(coh::Vector{Tr}, c_min::Vector{Tr}, c_max::Vector{Tr}, expV::Matrix{Tr},
    tol_GS::Tr, u::Function, mallaA::Vector{Tr}) where {Tr<:Real}

    # Weights
    α1 = (3-sqrt(5))/2
    α2 = (sqrt(5)-1)/2
    
    # Auxiliary variables
    d = c_max - c_min   # Distance
    siz = size(expV,1)  # Vector length
    tst_GS = 1          # convergence criterion
    
    # Initialise variables
    
    ## FIRST GUESS
    c1 = c_min + α1*d   # Guess for consumption
    u1 = u.(c1)         # Utility
    a1 = coh - c1       # Implied savings
    # Interpolate expected continuation value
    indL1, indU1, wgt1 = getWeights(a1, mallaA)
    ev1 = ((one(Tr) .- wgt1) .* expV[(1:siz) + (indU1 .- 1)*siz]  # upper neighbour
                    + wgt1  .* expV[(1:siz) + (indL1 .- 1)*siz]) # lower neighbour
    # Value function
    v1 = u1 + ev1
    # REMARK:   β does not appear in this formula because expV should have already
    #           been multiplied by β.
    
    ## SECOND GUESS (analogous)
    c2 = c_min + α2*d    # Guess for consumption
    u2 = u.(c2)         # Utility
    a2 = coh - c2       # Implied savings
    # Interpolate expected continuation value
    indL2, indU2, wgt2 = getWeights(a2, mallaA)
    ev2 = ((one(Tr) .- wgt2) .* expV[(1:siz) + (indU2 .- 1)*siz]  # upper neighbour
                    + wgt2  .* expV[(1:siz) + (indL2 .- 1)*siz]) # lower neighbour
    # Value function
    v2 = u2 + ev2
    # REMARK:   β does not appear in this formula because expV should have already
    #           been multiplied by β.
    
    iter = 0

    ## GOLDEN SEARCH
    d = α1*α2*d # update distance
    while tst_GS>tol_GS
        iter+=1
        # Some nodes are already solved. Focus on the others
        iGS = abs.(d) .> tol_GS
        # Update distance
        d = d*α2
    
        # if v2<v1:
            # Indicator: 1  in positions where f1>f2
            iHigh = v1 .> v2
            iChange = (iHigh .& iGS)
            # c2 is the new upper bound
            c2[iChange]     = c1[iChange]
            a2[iChange]     = a1[iChange]
            v2[iChange]     = v1[iChange]
            indL2[iChange]  = indL1[iChange]
            # Update lower bound
            c1[iChange] = c1[iChange]-d[iChange]
            u1[iChange] = u.(c1[iChange])               # current utility
            a1[iChange] = coh[iChange] - c1[iChange]    # savings
            # Interpolate expected continuation value
            indAux = (1:siz)[findall(iChange)]
            indL1[iChange], indU1, wgt1 = getWeights(a1[iChange], mallaA)
            ev1[iChange] = ((one(Tr) .- wgt1) .* expV[indAux + (indU1 .- 1)*siz]
                                      + wgt1  .* expV[indAux + (indL1[iChange] .- 1)*siz])
            # Value of first guess
            v1[iChange] = u1[iChange] + ev1[iChange]
    
        # else, f2>f1:
            # Update indicator: 1  in positions where f2>f1
            iHigh = .!iHigh
            iChange = iHigh .& iGS
            # c1 is the new lower bound
            c1[iChange]     = c2[iChange]
            a1[iChange]     = a2[iChange]
            v1[iChange]     = v2[iChange]
            indL1[iChange]  = indL2[iChange]
            # Update upper bound
            c2[iChange] = c2[iChange] + d[iChange]
            u2[iChange] = u.(c2[iChange])               # current utility
            a2[iChange] = coh[iChange] - c2[iChange]    # savings
            # Interpolate expected continuation value
            indAux = (1:siz)[findall(iChange)]
            indL2[iChange], indU2, wgt2 = getWeights(a2[iChange], mallaA)
            ev2[iChange] = ((one(Tr) .- wgt2) .* expV[indAux + (indU2 .- 1)*siz]
                                     + wgt2  .* expV[indAux + (indL2[iChange] .- 1)*siz])
            # Value of the second guess
            v2[iChange] = u2[iChange] + ev2[iChange]

        # Update convergence criterion
        tst_GS = maximum(abs.(d))
    end
    
    ## RETURNS
    
    # Return the larger of the two
    iHigh = v2.>v1
    c1[iHigh]       = c2[iHigh]
    a1[iHigh]       = a2[iHigh]
    v1[iHigh]       = v2[iHigh]
    indL1[iHigh]    = indL2[iHigh]

    return c1, a1, v1, indL1
    
end



##########################################################################
#### ENDOGENOUS GRID METHOD (not used)                                ####
##########################################################################

# not using it due to domain problems (negative consumption in some
# states of the oldest generation)

function EGM_savings(r::Tr, cnext::Vector{Tr}, anext::Vector{Tr}, inc_l::Vector{Tr},
    Πtrans::SparseMatrixCSC{Tr,Ti}, idioZ::Vector{Ti},
    eco::Economia, her::Herramientas
    ) where {Tr<:Real, Ti<:Integer}
    @unpack β, u′, inv_u′, a_min = eco
    @unpack n, a_max = her

    # Auxiliary variables
    siz = length(cnext)

    # Initialise policy function
    a_EGM = Array{Float64}(undef, siz)

    # Implied consumption (Euler equation)
    c_imp = reshape(inv_u′.(β*(1+r)*reshape(u′.(cnext),n.a,:) * Πtrans'), siz)

    # Implied current savings (budget constraint)
    a_imp = (anext + c_imp - inc_l) / (1+r)

    # Invert to get next-period-savings policy function
    for zz=1:n.z
        indZ = idioZ.==zz
        a_EGM[indZ] .= interpLinear(mallaA, a_imp[indZ], mallaA; metodo="cap");
    end

    # Borrowing constraint
    clamp.(a_EGM, a_min, a_max)
    # Impose upper bound as well, otherwise error when getting weights for the measure

    return a_EGM
end



##########################################################################
#### HOUSEHOLD PROBLEM                                                ####
##########################################################################


function hh_solve!(eco::Economia, her::Herramientas, sol::Solucion, cfg::Configuracion)::Nothing
    
    # Unpack relevant variables
    @unpack β, u, a_min = eco;
    @unpack n, mallaA, a_max, mallaZ, mallaζ, redQ, redSt, matSt, id, ind = her;
    @unpack r, w, inc_l, inc_a = sol;
    @unpack tol_GS, c_min, penal = cfg;

    # Types
    Tr = typeof(r)
    Ti = typeof(n.j)

    # Initialise variables
    a_GS = Array{Tr}(undef, n.N)    # golden search savings policy function
    c_GS = similar(a_GS)            # golden search consumption policy function
    v_GS = similar(a_GS)            # golden search value function
    apos_GS = Array{Ti}(undef, n.N) # lower position of a_GS in assets grid
    
    # Cash on hand
    coh = inc_l + inc_a + mallaA[matSt[:,id.a]]

    # Oldest generation: no savings
    a_GS[ind.last] .= 0.0
    c_GS[ind.last] .= coh[ind.last]
        # Impose floor on consumption
        indCmin = c_GS .< c_min
        replace!(c -> c < c_min ? c_min : c, c_GS)
        v_GS[ind.last] .= u.(c_GS[ind.last])
        v_GS[ind.last .& indCmin] .= penal
        apos_GS[ind.last] .= id.a0

    # Other generations
    for jj=range(; start=n.j-1, stop=1, step=-1)
        # Generation indicators
        indJ = matSt[:,id.j].==jj
        indJnext = redSt[:,id.j].==(jj+1)
        # Endogenous grid method
            # # Labour income
            # inc_l = w*mallaζ[matSt[indJ,id.j]].*mallaZ[matSt[indJ,id.z]]
            # # Next period consumption
            # cnext = c_EGM[matSt[:,id.j].==(jj+1)]
            # # Policy functions
            # a_EGM[indJ] = EGM_savings(r,
            #                           cnext,                      # next period consumption
            #                           mallaA[matSt[indJ,id.a]],   # next period savings
            #                           inc_l,                      # labour income
            #                           redQ[indJ,indJnext],        # state transition probabilities
            #                           matSt[indJ, id.z],          # productivity state
            #                           eco, her)
            # c_EGM[indJ] = inc_l .+ (1+r)*mallaA[matSt[indJ,id.a]] .- a_EGM[indJ]
        # Golden search
            # Expected continuation value
            vnext = v_GS[matSt[:,id.j].==(jj+1)]
            expV = β * redQ[indJ,indJnext] * reshape(vnext, n.aj[jj+1], :)'
            # Golden search bounds
            cmax_GS = max.(c_min, coh[indJ] .- a_min)
            cmin_GS = max.(c_min, coh[indJ] .- mallaA[n.aj[jj+1]])
            # Optimisation
            c_GS[indJ], a_GS[indJ], v_GS[indJ], apos_GS[indJ] = golden_policies(coh[indJ], cmin_GS, cmax_GS, expV, tol_GS, u, mallaA[1:n.aj[jj+1]])
    end
    
    println("Done with Golden search.")

    # Update solution
    sol.a_pol  .= a_GS
    sol.c_pol  .= c_GS
    sol.value  .= v_GS
    sol.apos_L .= apos_GS

    return nothing
end



##########################################################################
#### Q-TRANSITION MATRIX                                              ####
##########################################################################

function transitionMat( a_pol::Vector{Tr}, apos_L::Vector{Ti}, as::Vector{Tr},
    st_a::Vector{Ti}, st_j::Vector{Ti}, redtrans::SparseMatrixCSC{Tr,Ti},
    redst_j::Vector{Ti}; jmax::Ti=maximum(st_j)
  )::SparseMatrixCSC{Tr,Ti} where {Tr<:Real,Ti<:Integer}
    
    # Auxiliary variables
    na = size(as,1)     # number of points in grid for decision variable
    nN = size(redtrans,1)   # number of states

    # Initialise variable
    trans = spzeros(Tr,Ti,nN,nN)

    # Position in assets grid
    apos_U, wgt = getWeights(a_pol, as, apos_L; metodo="cap")
    
    # Savings in next period
    anext = spzeros(Tr,Ti,na,nN)    # Initialise variable
    anext[apos_L + ((1:nN) .- 1)*na] .= wgt
    anext[apos_U + ((1:nN) .- 1)*na] .= (1.0 .- wgt)

    # Complete the matrix generation by generation
    for st = findall(st_j.<jmax)
        jj = st_j[st]
        rowsL = (st_a.==apos_L[st]) .& (st_j.==jj+1)
        rowsU = (st_a.==apos_U[st]) .& (st_j.==jj+1)
        trans[rowsL, st] .= redtrans[st, redst_j.==(jj+1)] * wgt[st]
        trans[rowsU, st] .= redtrans[st, redst_j.==(jj+1)] * (one(Tr) - wgt[st])
    end
    return trans
end

function Q_matrix(sol::Solucion, her::Herramientas)::SparseMatrixCSC
    @unpack a_pol, apos_L = sol
    @unpack n, mallaA, matSt, id, ind, redQ, redSt = her
    
    # Compute transition matrix
    Q_mat = transitionMat(a_pol, apos_L, mallaA, matSt[:,id.a], matSt[:,id.j], redQ, redSt[:,id.j])

    # Correction: new generation
    Q_mat[ind.newby, ind.newpt] .= 0
    Q_mat[ind.newby .& (matSt[:,id.a].==id.a0), ind.newpt] .= redQ[ind.newpt,redSt[:,id.j].==1]'

    # Return matrix
    println("Done with Q-matrix.")
    return Q_mat
end;



##########################################################################
#### STATIONARY DISTRIBUTION                                          ####
##########################################################################

# Initialise distribution
# Guess, consistent with long-run distribtion of age and productivity
function guessDist(jRet::Ti,her::Herramientas{Tr,Ti})::Vector{Tr} where {Tr<:Real, Ti<:Integer}
    @unpack matSt, id, n, Sz, ind = her;
    # Initialise distribution
    distr = Vector{Tr}(undef,n.N)
    # Mass of agents in each age group:
        # Mass of agents by age
        jMass = fill(1/n.j, n.j)
    # Mass of agents in each productivity group:
        zMass = Sz'
    # Iterate over groups of individuals with same age and productivity
    for jj=1:(jRet-1), zz=1:n.z
        indJZ = (matSt[:,id.j].==jj) .& (matSt[:,id.z].==zz)    # Auxiliary indicator
        distr[indJZ] .= jMass[jj]*zMass[zz]/sum(indJZ)
    end
    # Productivity does not matter after retirement
    for jj=jRet:n.j
        indJ = (matSt[:,id.j].==jj)    # Auxiliary indicator
        distr[indJ] .= jMass[jj]/sum(indJ)
    end
    # Comprobaciones
    # jzMass = vcat([[sum(distr[(zz .∈ matSt[:,id.z]) .& (jj .∈ matSt[:,id.j])]) for zz in 1:n.z]' for jj in 1:n.j]...)
    # @assert all(sum(jzMass[1:(jRet-1),:]./sum(jzMass[1:(jRet-1),:]),dims=1).≈zMass)
    # @assert all(sum(jzMass,dims=2).≈jMass)

    return distr
end

# Eigenvalues
function dist(M::SparseMatrixCSC{Tr,Ti}, tol::Tr, maxiter::Ti
    )::Vector{Tr} where {Tr<:Real,Ti<:Integer}
    nM = size(M)[1]
    _, x = powm!(Matrix(M), ones(Tr, nM), maxiter=maxiter, tol=tol)
    # returns the approximate largest eigenvalue λ of M and one of its eigenvector
    return x / sum(x)
end

# Iterative method
function dist(distr0::Vector{Tr}, Q_mat::SparseMatrixCSC, tol::Tr, maxit::Ti
              )::Vector{Tr} where {Tr<:Real, Ti<:Integer}
    # Stationary distribution
    tst = one(Tr)   # Initialise convergence criterion
    for iter=1:maxit
        distr1 = Q_mat*distr0
        tst = maximum(abs.(distr1 - distr0))
        # If it converged
        if tst<tol
            println("Done with stationary distribution.")
            return distr0
            break
        end
        distr0 .= distr1
    end
    # If the loop finished without converging
    error("Distribution did not converge")
end;



##########################################################################
#### STEADY STATE COMPUTATION                                         ####
##########################################################################

function guessV(eco::Economia{Tr,Ti}, her::Herramientas, sol::Solucion,
    cfg::Configuracion)::Vector{Tr} where {Tr<:Real, Ti<:Integer}

    @unpack β, γ = eco;
    @unpack matSt, id, ind, mallaA = her;
    @unpack inc_l, inc_a = sol;
    @unpack c_min, penal = cfg;

    # Guess for value function
    v0 = 1/(1-β) * ((max.(c_min, inc_l + inc_a)).^(1-γ))/(1-γ)

    # Penalty for negative assets when there is death risk
    iPunish = ind.ret .& (matSt[:,id.a] .< id.a0)
    v0[iPunish] = penal * (-mallaA[matSt[iPunish,id.a]]) 

    return v0
end

function steady(eco::Economia, her::Herramientas, cfg::Configuracion;
    r_0::Tr=0.04)::Solucion where {Tr<:Real}
    
    @unpack β, α, δ, u′, a_min = eco;
    @unpack n, mallaA, mallaZ, mallaζ, matSt, id = her;
    @unpack rlx_SGE, tol_SGE, maxit_SGE, c_min = cfg;

    # We initialize the solution
    sol = Solucion(r_0, c_min, eco, her);
    sol.value = guessV(eco,her,sol,cfg)     # Guess for value function (relevant for VFI and donation)

    # Auxiliary variables
    st_ζ = mallaζ[matSt[:,id.j]]
    st_z = mallaZ[matSt[:,id.z]]
    st_a = mallaA[matSt[:,id.a]]

    # Start of the dichotomy
    for kR = 1:maxit_SGE
        # Update prices
        sol_update_r!(r_0, st_ζ, st_z, st_a, eco, sol)
        # Policy functions (Golden)
        hh_solve!(eco, her, sol, cfg)
        # Q-matrix
        sol.Q_mat = Q_matrix(sol, her)
        # Measure
        sol_update_distr!(sol, cfg.tol_SSdis, cfg.maxit_SSdis, eco, her)
        # Aggregate quantities
        sol_update_agg!(sol)
        @unpack A_agg, K_agg, L_agg = sol;
        # If it converged
        if abs(K_agg - A_agg) < tol_SGE
            # The dichotomoy has converged: we fill the solution structure
            println("Markets clear! in: ", kR, " iterations\n")
            # Returning the result (successful case)
            return sol
            break
        end
        # If it did not converge yet
            # Implied r
            r_1 = α*(A_agg/L_agg)^(α-1) - δ
            # Display current situation
            println("#", kR, ", ir: ", fmt.(100 * ([r_0, r_1]), 6), ", supply: ", fmt(A_agg),
                ", demand-supply gap: ", K_agg - A_agg, ".\n")
            flush(stdout)
            # Update guess (relaxation)
            r_0 = rlx_SGE*r_0 + (1-rlx_SGE)*r_1
    end
    println("Markets did not clear")
    # Returning the result (failure case)
    return sol
end;



##########################################################################
#### CHECK SOLUTION                                                   ####
##########################################################################

function check_solution(eco::Economia, her::Herramientas, sol::Solucion,
    tol::Tr)::Bool where {Tr<:Real}
    @unpack β, α, δ, u′, a_min = eco;
    @unpack n, mallaA, mallaZ, matSt, id, ind = her;
    @unpack a_pol, c_pol, r, w, A_agg, K_agg, C_agg, L_agg, Q_mat, distr = sol;
    
    ## Q TRANSITION MATRIX
    if !all(sum(Q_mat,dims=1)' .≈ 1.0 .+ ind.newpt - ind.last)
        println("Q matrix not consistent with survival probabilities: ",
            (.!(sum(Q_mat,dims=1)' .≈ 1.0 .+ ind.newpt - ind.last)) .* (1:n.N))
        return false
    end

    ## DISTRIBUTION
    # General
    if !(sum(distr) ≈ 1.0)
        println("error in stationary distribution: ",
            sum(distr))
        return false
    end
    # By age
    jDistr = vcat([sum(distr[jj .∈ matSt[:,id.j],:], dims=1) for jj in 1:n.j]...)
        # Auxiliary variables
        auxMass = jDistr[1]         # mass of agents by age (before death risk)
    if !all(auxMass .≈ jDistr)
        println("error in survival probabilities")
        return false
    end

    ## SAVINGS
    if !(L_agg * ((r + δ) / α)^(1.0 / (α - 1.0)) ≈ K_agg)
        println("error in aggregate savings: ", L_agg * ((r + δ) / α)^(1.0 / (α - 1.0)) - K_agg)
        return false
    end
    if !(A_agg ≈ sum(distr .* a_pol))
        println("error in aggregate savings: ", A_agg, sum(distr .* a_pol))
        return false
    end
    if abs(A_agg - sum(distr .* mallaA[matSt[:,id.a]])) > n.N * tol
        println("difference in savings: ", "\n",
            A_agg, "\n", sum(distr .* mallaA[matSt[:,id.a]]), "\n",
            A_agg - sum(distr .* mallaA[matSt[:,id.a]]))
        return false
    end

    ## CONSUMPTION
    C_ = w*L_agg + (1+r) * sum(distr .* mallaA[matSt[:,id.a]]) - A_agg
    if  abs(C_agg - C_) > n.N * tol
        println("error in aggregate consumption:\n", C_, "\n", C_agg, "\n", C_ - C_agg)
        return false
    end

    return true
end;



##########################################################################
#### INTERPRETING THE SOLUTION                                        ####
##########################################################################

function SS_annual(eco::Economia, her::Herramientas, sol::Solucion)
    nt = her.n.t
    # Initialise annualised structure
    sol_annual = Solucion(sol.r, sol.w, eco, her, copy(sol.a_pol), copy(sol.c_pol), copy(sol.value), copy(sol.apos_L), copy(sol.Q_mat), copy(sol.distr));
    # Annualise flow variables
    sol_annual.c_pol .= sol.c_pol/nt
    sol_annual.inc_l .= sol.inc_l/nt
    sol_annual.inc_a .= sol.inc_a/nt
    sol_annual.Y_agg = sol.Y_agg/nt
    sol_annual.C_agg = sol.C_agg/nt
    sol_annual.r     = (1+sol.r)^(1/nt)-1
    # Return annualised solution
    return sol_annual
end

function SS_scale!(actual_mean_gr_inc::Tr, sol::Solucion)::Tr where {Tr<:Real}
    @unpack inc_l, inc_a, distr = sol

    # This function scales the variables in "sol" and returns the conversion factor used

    # Gross income
        # At household level
        gr_inc = inc_l + inc_a
        # On average
        model_mean_gr_inc = sum(gr_inc.*distr) ./ sum(distr)
        # Conversion factor
        conv = actual_mean_gr_inc / model_mean_gr_inc

    # Scale variables
    sol.a_pol .*= conv
    sol.c_pol .*= conv
    sol.inc_l .*= conv
    sol.inc_a .*= conv
    sol.A_agg *= conv
    sol.C_agg *= conv
    sol.K_agg *= conv
    sol.L_agg *= conv
    sol.Y_agg *= conv

    return conv
end