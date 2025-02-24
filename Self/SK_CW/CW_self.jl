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
end


# self equation
function sce(β, m)
    tanh(β*m)
end


# fixed point iteration 
function solve(β) 
    m = 1.0
    for i = 1:1000
        nm = sce(β, m)
        m = (nm+m)/2
        if abs(m-nm) < 10^-10
            break
        end
    end
    m
end

# temperature range
T = LinRange(0.0001,2,100)

# for all t in T we evaluete the fixed point solution of the self equation
mags = @showprogress[solve(1/t) for t in T]

# same thing but with static array
#mags = solve.(1 ./ T)

#plot of the results
let
    cla()
    plot(T,mags)
    title(L"\bar{m}=\tanh[\beta\bar{m}]")
    xlim(0,2)
    ylim(-0.05,1.05)
    xlabel(L"\beta^{-1}", size = 12)
    ylabel(L"\bar{m}(\beta)", size = 12)
    #savefig("/CW.png")
end
tight_layout()