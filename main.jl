# Solving OLG version of Aiyagari model with:
# - Exogenous labour supply
# - Retirement
# - Certain lifespan
# Starting point: codes by Le Grand & Ragot

# Load packages
using BenchmarkTools
using IterativeSolvers  # powm!
using LinearAlgebra     # dot
using Parameters        # @unpack
using StatsPlots
using QuantEcon         # rouwenhorst
using Roots             # findzero
using SparseArrays      # SparseMatrixCSC
# using TimerOutputs
# tmr = TimerOutput()

# Include other files
include("Funciones.jl");
include("Structures.jl");
include("SS_solve.jl");
include("SS_display.jl");

# Initialise model
    # eco: model parameters and functions
    # her: tools to solve it (grids, probabilities, state matrix, etc.)
    eco1, her1 = Model(50, 100, 5, -0.3, 10.0; β=0.97, δ=0.025, α=0.3, ρz=0.9, σz=0.2); # nj=16, γ=1.0001, δ=0.025, nt=5, α=0.36, β=0.999, ρz=0.9, σz=0.2);

# Configuration for numerical solution
    cfg1 = Configuracion(0.5; doubBin=true);

# Find steady state equilibrium
    sol1 = steady(eco1, her1, cfg1; r_0=0.08);
    # Verifications
    @assert check_solution(eco1, her1, sol1, cfg1.tol_check)

# Facilitate the interpretation
    # Annualise solution
    sol_annual = SS_annual(eco1, her1, sol1);
    # Scale it to an actual economy
    mean_gr_inc_FR = 34382.12   # From HFCS
    conv_FR = SS_scale!(mean_gr_inc_FR, sol_annual);

# Display results
    # Summary table
    (print_dict ∘ SS_summary)(conv_FR, eco1, her1, sol_annual)
    # Graphs
    SS_graphs(conv_FR, eco1, her1, sol_annual, cfg1)
    # Tables
    SS_tables(her1, sol_annual; nquants=5)