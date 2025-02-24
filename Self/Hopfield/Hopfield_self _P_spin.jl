begin
    using LinearAlgebra, PyPlot, ProgressMeter
    using Distributions
    using SpecialFunctions
    using DelimitedFiles
    using StatsBase
    using Random
    using StaticArrays
    using Statistics
    using JLD2

    function figsize(full,iϕ =0.6180469715698392)
        width = 3.487*ifelse(full,2.0,1.0)
        height = width*iϕ
        (width,height)
    end
    function init_pyplot()
        rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
        rcParams["text.usetex"] = true
        rcParams["figure.dpi"] = 200
        rcParams["figure.figsize"]          =   figsize(false)
        rcParams["path.simplify"]           =   true
        rcParams["font.family"]             =   "serif"
        rcParams["mathtext.fontset"]        =   "custom"
        rcParams["xtick.major.size"]        =   5
        rcParams["ytick.major.size"]        =   5
        rcParams["xtick.minor.size"]        =   2.5
        rcParams["ytick.minor.size"]        =   2.5
        rcParams["xtick.major.width"]       =   0.8
        rcParams["ytick.major.width"]       =   0.8
        rcParams["xtick.minor.width"]       =   0.8
        rcParams["ytick.minor.width"]       =   0.8
        rcParams["lines.markeredgewidth"]   =   0.8
        rcParams["legend.numpoints"]        =   1
        rcParams["legend.frameon"]          =   true
        rcParams["legend.handletextpad"]    =   0.3
        rcParams["xtick.labelsize"] = 8
        rcParams["ytick.labelsize"] = 8
        rcParams["axes.labelsize"]  = 8
        rcParams["legend.fontsize"] = 7
        rcParams["legend.title_fontsize"] = 7
        rcParams["lines.linewidth"] = 1.0
        rcParams["lines.markersize"] = 3.0
        pygui(true)
    end
    init_pyplot()
end



#gaussian integration
function gaussian(f)
    # Gaussian weight
    W(x) = exp(- x^2 / 2) / sqrt(2 * pi)
    # discretization of the intervals
    rg = LinRange(-16.0, 16.0, 2^13)
    # function × Gaussian weight
    int = f(rg[1]).*W(rg[1])
    # normalization factor
    norm = W(rg[1])
    # for cicle over each discretized piece
    for i = 2:length(rg)
        x = rg[i]
        wh = W(x)
        int = int .+ f(x).*wh
        norm += wh
    end
    return int./norm
end

####################################
####################################
# EXAMPLE
gaussian(x-> [tanh(x),tanh(x)^2])

gaussian() do x
    [tanh(x), tanh(x)^2]
end
# real value = 0.39429449039746706
####################################
####################################



# self-cons equations
function sce(β, γ, m, q, P)
    if P > 2
        p = (P/2)*q^(P-1)
    else
        p = q/(1-β*(1-q))^2
    end
    signal = β*(P/2)*m^(P-1)
    noise = β*sqrt(γ*p)
    gaussian() do x
        t = tanh(signal + x * noise)
        @SVector[t,t^2]
    end
end


# fixed point iteration 
function solve(β, γ, m, q, P) 
    for i = 1:2000
        nm, nq = sce(β,γ, m, q, P)
        m, q = ([nm, nq] .+ [m,q])./2
        if norm([m,q] - [nm, nq]) < 1e-12
            break
        end
    end
    m,q
end

# T = 0.2, γ= 0.01, R region (m,q)= (>0,>0)
solve(1/0.2, 0.01, 1, 1, 4) 
# T = 0.2, γ= 0.5, SG region (m,q)= (0, >0)
solve(1/0.2, 0.5, 1, 1, 4) 
# T = 1.0, γ= 0.5, E region (m,q)= (0,0)
solve(1/0.999, 0.0, 1, 1, 4) 




# retrieval bisection which takes a value of T and compute the relative values of the load
function bisection(T, P, start_γ)
    β=1/T
    # starting load and jump
    γ, dγ = [start_γ, start_γ/2]
    while dγ>0.001
        # using the fixed point function for fixed value of γ and β we compute the value of m and q
        m, q = solve(β, γ, 1.0, 1.0, P)
        # we check tha m value
        # if m < 0.001 we are in E-region, thus γ → γ - dγ
        if m < 0.01 
            γ -= dγ
        # if m > 0.001 we are in R-region, thus γ → γ + dγ
        else
            γ += dγ
        end
        # after each iteration we halve the value of dγ
        dγ/=2
    end
    # we return the critical load at input temperature
    γ
end



# SG bisection which takes a value of γ and compute the relative values of the temperature
function bisection_erg(γ, P, starting_T)
    # starting temperature and jump
    T, dT = [starting_T, starting_T/2]
    while dT>0.0001
        β=1/T
        # using the fixed point function for fixed value of γ and β we compute the value of m and q
        m, q = solve(β, γ, 0.0, 1.0, P)
        # we check tha q value
        # if q < 0.001 we are in E-region, thus T → T - dT
        if q < 0.001 
            T -= dT
        # if q > 0.001 we are in SG-region, thus T → T + dT
        else
            T += dT
        end
        # after each iteration we halve the value of dT
        dT/=2
    end
    # we return the critical temperature at input load
    T
end



P =4
T = LinRange(0.001, 2, 40)
Aerg = LinRange(0.0, 1, 40)


A = @showprogress[bisection(t, P, 0.5) for t in T]
Terg = @showprogress[bisection_erg(a, P,1) for a in Aerg]


let
    cla()
    title("Dense Hopfield model with P="*string(P))
    #plot R-SG region
    plot(A,T, label = L"R\to SG")
    # plot SG-E region
    plot(Aerg,Terg, label = L"SG\to E")

    xlim(0,1)
    ylim(0,2)
    ylabel(L"\beta^{-1}", size =12)
    xlabel(L"\gamma", size =12)
    legend(loc="lower right", fontsize = 10)
end
tight_layout()