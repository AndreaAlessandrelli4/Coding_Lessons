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
    rcParams["legend.frameon"]          =   false
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
















function gaussian(f)
    # Gaussian weight
    W(x) = exp(- x^2 / 2) / sqrt(2 * pi)
    # discretization of the intervals
    rg = LinRange(-16.0,16.0,4096)
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


gaussian(x-> [tanh(x),tanh(x)^2])

gaussian() do x
    [tanh(x), tanh(x)^2]
end
# real value = 0.39429449039746706
    

# self-cons equations
function sce(β, γ, m, q)
    p = q/(1-β*(1-q))^2
    signal = β*m
    noise = β * sqrt(γ * p)
    gaussian() do x
        t = tanh(signal + x*noise)
        [t,t^2]
    end
end
sce(1, 1, 1, 1)


# fixed point iteration 
function solve(β, γ, m, q) 
    for i = 1:1000
        nm, nq = sce(β, γ, m, q)
        m, q = ([nm, nq] .+ [m,q])./2
        if norm([m,q] - [nm, nq]) < 1e-10
            break
        end
    end
    m,q
end

# T = 1.5, α= 0.0, E region (m,q)= (0, 0)
solve(1/1.5, 0.0, 1, 1) 
# T = 0.5, α= 0.01, R region (m,q)= (>0 , >0)
solve(1/0.5, 0.01, 1, 1) 
# T = 0.5, α= 0.5, SG region (m,q)= (0, >0)
solve(1/0.5, 0.3, 0, 1) 





# Create an array of length=Tlen of temperatures which starts from 0.005 and ends at 1.5
Tlen = 50 
T = LinRange(0.005, 1.5, Tlen)
# Create an array of length=Alen of loads which starts from 0.0 and ends at 0.3
Alen = 50 
A = LinRange(0.0, 0.3, Alen)


# grid of T × A
Tmat = [t for t in T, a in A]
Amat = [a for t in T, a in A]
RETmat = @showprogress[solve(1 / t, a, 1.0, 1.0) for t in T, a in A]


let
    cla()
    th = 0.005
    # get the first index of the tuple for each grid element (where we have stored the magn value)
    mag = getindex.(RETmat,1)
    # get the second index of the tuple for each grid element (where we have stored the overlap value)
    sg = getindex.(RETmat,2)

    # drawing of demarcation lines
    contour(Amat,Tmat, mag, levels=[sqrt(0.005)], colors=["black"])
    contour(Amat,Tmat, sg, levels=[0.005], colors=["red"])

    # drawing of the value of the mag or the overlap inside each grid point
    pcolormesh(Amat.+0.001,Tmat, mag, cmap="Blues")
    #pcolormesh(Amat.+0.001,Tmat, sg, cmap="Reds")
    colorbar()

    # denomination of the various region
    text(0.025,0.25,"R",color="white",fontsize=15)
    text(0.16,0.25,"SG",color="black",fontsize=15)
    text(0.025,1.34,"P",color="black",fontsize=15)

    #
    ylabel(L"\beta^{-1}", size =12)
    xlabel(L"\gamma", size =12)
end
tight_layout()


# to save
savefig("/Users/andreaalessandrelli/Desktop/Sapienza_Lesson/Seld_Julia/Hopfield/Hopfield.png")









using StaticArrays
using Random

x = [1,3,-3,0]
y = [6,0,-4,-1]

transpose(x) * y
z = x .* y

