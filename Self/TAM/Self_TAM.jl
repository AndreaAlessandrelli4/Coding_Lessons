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
    rg = LinRange(-16.0,16.0,4096*4)
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
# SELF OF TAM
function detC_dev_sigma(a,b,c,qsigma,qtau,qphi,β)
    2*(1-qphi)*(1-qtau)*β^3*(a*b*c)+ (1-qphi)*β^2*b^2+(1-qtau)*β^2*a^2
end


function detC_dev_tau(a,b,c,qsigma,qtau,qphi,β)
    2*(1-qphi)*(1-qsigma)*β^3*(a*b*c)+ (1-qphi)*β^2*c^2+(1-qsigma)*β^2*a^2
end

function detC_dev_phi(a,b,c,qsigma,qtau,qphi,β)
    2*(1-qsigma)*(1-qtau)*β^3*(a*b*c)+ (1-qsigma)*β^2*b^2+(1-qtau)*β^2*c^2
end

function detC(a,b,c,qsigma,qtau,qphi,β)
    1 - ((1-qtau)*(1-qsigma)*β^2*a^2+(1-qphi)*(1-qtau)*β^2*c^2+(1-qphi)*(1-qsigma)*β^2*b^2)-2*(1-qphi)*(1-qtau)*(1-qsigma)*β^3*(a*b*c)
end



function numer(a,b,c,γ,qx,qy,qz,beta)
    primo =beta^3*(a*b*c)*(qx*(1-qy)*(1-qz)+qy*(1-qx)*(1-qz)+qz*(1-qy)*(1-qx))
    secondo = beta^2*a^2*qy*(1-qx)+beta^2*a^2*qx*(1-qy)+beta^2*c^2*qy*(1-qz)+beta^2*c^2*qz*(1-qy)+beta^2*b^2*qx*(1-qz)+beta^2*b^2*qz*(1-qx)
    γ * primo + γ*secondo/(2)
end




function beta_sigma(a,b,c,γ,qsigma,qtau,qphi,beta)
    determinanteC = detC(a,b,c,qsigma,qtau,qphi,beta)
    detC_deriv = detC_dev_sigma(a,b,c,qsigma,qtau,qphi,beta)
    numeratore = numer(a,b,c,γ,qsigma,qtau,qphi,beta)

    primo = (-1)* γ/2 * detC_deriv/determinanteC
    secondo = (-1)*numeratore * detC_deriv/(determinanteC)^2

    terzo = γ* beta^3*(a*b*c)*( (1-qtau)*(1-qphi) -qphi*(1-qtau)-qtau*(1-qphi) )/determinanteC
    quarto = γ*( beta^2*a^2*(1-qtau) - beta^2*a^2*qtau -beta^2*b^2*qphi + beta^2*b^2*(1-qphi))/(2*determinanteC)

    (primo +secondo +terzo+quarto)*(-1)*(2)
end

function beta_tau(a,b,c,γ,qsigma,qtau,qphi,beta)
    determinanteC = detC(a,b,c,qsigma,qtau,qphi,beta)
    detC_deriv = detC_dev_tau(a,b,c,qsigma,qtau,qphi,beta)
    numeratore = numer(a,b,c,γ,qsigma,qtau,qphi,beta)

    primo = (-1)* γ/2 * detC_deriv/determinanteC
    secondo = (-1)*numeratore * detC_deriv/(determinanteC)^2

    terzo = γ* beta^3*(a*b*c)*( (1-qsigma)*(1-qphi) -qphi*(1-qsigma)-qsigma*(1-qphi) )/determinanteC
    quarto = γ*( beta^2*a^2*(1-qsigma) - beta^2*a^2*qsigma -beta^2*c^2*qphi + beta^2*c^2*(1-qphi))/(2*determinanteC)

    (primo +secondo +terzo+quarto)*(-1)*(2)
end


function beta_phi(a,b,c,γ,qsigma,qtau,qphi,beta)
    determinanteC = detC(a,b,c,qsigma,qtau,qphi,beta)
    detC_deriv = detC_dev_phi(a,b,c,qsigma,qtau,qphi,beta)
    numeratore = numer(a,b,c,γ,qsigma,qtau,qphi,beta)

    primo = (-1)* γ/2 * detC_deriv/determinanteC
    secondo = (-1)*numeratore * detC_deriv/(determinanteC)^2

    terzo = γ* beta^3*(a*b*c)*( (1-qtau)*(1-qsigma) -qsigma*(1-qtau)-qtau*(1-qsigma) )/determinanteC
    quarto = γ*( beta^2*c^2*(1-qtau) - beta^2*c^2*qtau -beta^2*b^2*qsigma + beta^2*b^2*(1-qsigma))/(2*determinanteC)

    (primo +secondo +terzo+quarto)*(-1)*(2)
end

############################## RETRIEVAL  #################################
function sce(a, b, c, γ, α, θ, β, m, n, l, qm, qn, ql)
    s_m = β * (n * a/θ + l* b/α)
    s_n = β * (m * a + l * c/α)
    s_l = β * (m * b + n * c/θ)

    n_m = abs(beta_sigma(a,b,c,γ,qm,qn,ql,β))
    n_n = abs(beta_tau(a,b,c,γ,qm,qn,ql,β))
    n_l = abs(beta_phi(a,b,c,γ,qm,qn,ql,β))

    
    m, n, l, qm, qn, ql = midpoint() do x
        arg_m =  (s_m + x * sqrt(n_m) )
        arg_n = θ * (s_n + x * sqrt(n_n) )
        arg_l = α * (s_l + x * sqrt(n_l) )
        tm = tanh( arg_m )
        tn = tanh( arg_n )
        tl = tanh( arg_l )
        @SVector [tm, tn, tl, tm*tm, tn*tn, tl*tl]
    end
end


function fixed_point(a, b, c, γ, α, θ, β) 
    m, n, l=(1,1,1)
    qm, qn, ql=(1,1,1)
    delta = Inf
    maxsteps = 1000
    for i in 1:maxsteps
        nm, nn, nl, nqm, nqn, nql = sce(a, b, c, γ, α, θ, β, m, n, l, qm, qn, ql)
        delta = findmax([abs(nm-m), abs(nn-n),abs(nl-l),abs(nqm-qm),abs(nqn-qn),abs(nql-ql)])[1]
        if delta < 1e-10
            break
        end
        m=(m+nm)/2
        n=(n+nn)/2
        l=(l+nl)/2
        qm=(qm+nqm)/2
        qn=(qn+nqn)/2
        ql=(ql+nql)/2
    end
    m, n, l, qm, qn, ql
end



function bisection(a, b, c, α, θ, β) 
    γ = 0.25
    deltaγ= 0.125
    m=0
    while deltaγ > 0.0001
        m, n, l, qm, qn, ql=fixed_point(a, b, c, γ, α, θ, β)
        if n<0.01
            γ -= deltaγ
            deltaγ /= 2
        else
            γ += deltaγ
            deltaγ /= 2
        end
    end
    γ
end


function bisection_erg(a, b, c, γ, α, θ) 
    T = 2.0
    deltaT= 1.0
    qm=0
    while deltaT > 0.0001
        β = 1/T
        m, n, l, qm, qn, ql=fixed_point(a, b, c, γ, α, θ, β)
        if qm<0.01
            T -= deltaT
            deltaT /= 2
        else
            T += deltaT
            deltaT /= 2
        end
    end
    T
end




Tv1 = LinRange(0.001, 2.5, 20)
Av1 =@showprogress[bisection(1, 1,1, 1.0, 1.0, 1/t) for t in Tv1]
Av11 =@showprogress[bisection(1, 1,1, 1, 1.5, 1/t) for t in Tv1]
Av111 =@showprogress[bisection(1, 1,1, 1, 1, 1/t) for t in Tv1]
Av1_11 =@showprogress[bisection(1, -1,-1, 1, 1, 1/t) for t in Tv1]


Av_erg = LinRange(0.001, 0.5, 15)

Tv1_erg =@showprogress[bisection_erg(1, 0,0,t, 1, 1) for t in Av_erg]
Tv11_erg =@showprogress[bisection_erg(1, 0,1,t, 1, 1) for t in Av_erg]
Tv111_erg =@showprogress[bisection_erg(1, 1,1,t, 1, 1) for t in Av_erg]

begin
    cla()
    plot(Av1, Tv1, label= L"(g_{\sigma\tau},g_{\phi\sigma},g_{\phi\tau})=(1,0,0)", color="grey", alpha=0.5)
    plot(Av_erg, Tv1_erg, color="grey", alpha=0.5)
    plot(Av11, Tv1 , label= L"(g_{\sigma\tau},g_{\phi\sigma},g_{\phi\tau})=(1,1,0)", color="grey", alpha=1)
    plot(Av_erg, Tv11_erg , color="grey", alpha=1)
    plot(Av111, Tv1, label= L"(g_{\sigma\tau},g_{\phi\sigma},g_{\phi\tau})=(1,1,1)", color="black")
    plot(Av_erg, Tv111_erg,  color="black")
    xlim(0,0.5)
    ylim(0,3)
    ylabel(L"\beta^{-1}", size =10)
    xlabel(L"\gamma", size =10)
    legend()
end
tight_layout()


begin
    cla()
    plot(Av1, Tv1, label= L"(\alpha,\theta)=(1.5,1.5)", color="grey", alpha=0.5)
    plot(Av11, Tv1 , label= L"(\alpha,\theta)=(1.0,1.5)", color="grey", alpha=1)
    plot(Av111, Tv1, label= L"(\alpha,\theta)=(1.0,1.0)", color="black")
    xlim(0,0.4)
    ylim(0,2.25)
    ylabel(L"\beta^{-1}", size =10)
    xlabel(L"\gamma", size =10)
    legend()
end
tight_layout()


