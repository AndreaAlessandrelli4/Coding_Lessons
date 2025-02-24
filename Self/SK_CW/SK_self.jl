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


# gaussian integration
function gaussian(f)
    W(x) = exp(-x^2/2)
    rg = LinRange(-16.0,16.0,4096)
    int = f(rg[1]).*W(rg[1])
    norm = W(rg[1])
    for i = 2:length(rg)
        x = rg[i]
        wh = W(x)
        int = int .+ f(x).*wh
        norm += wh
    end
    int./norm
end



function sce(β, q)
    gaussian() do z
        t = tanh(β*z*sqrt(q))
        return t^2
    end
end



# fixed point iteration 
function solve(β) 
    q = 1.0
    for i = 1:1000
        nq = sce(β, q)
        q = (nq + q)/2
        if abs(nq-q) < 10^-10
            break
        end 
    end
    q
end


T = LinRange(0.0001,2,100)

Q = @showprogress[solve(1/t) for t in T]

#Q = solve.(1 ./ T)


let
    cla()
    plot(T,Q)
    title(L"\bar{q}=\langle\tanh^2[\beta x \sqrt{\bar{q}}]\rangle_x")
    xlabel(L"\beta^{-1}", size = 12)
    ylabel(L"\bar{q}(\beta)", size = 12)
    xlim(0,2)
    ylim(-0.01,1.01)
end
tight_layout()
#savefig("/SK.png")