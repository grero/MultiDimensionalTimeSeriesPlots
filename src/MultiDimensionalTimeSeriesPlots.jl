module MultiDimensionalTimeSeriesPlots
using Makie
using StatsBase
using LinearRegressionUtils
using MultivariateStats
using Colors
using LinearAlgebra

struct Angle{T<:Real}
    a::T
end

Base.sin(a::Angle{T}) where T <: Real = sin(a.a)
Base.cos(a::Angle{T}) where T <: Real = cos(a.a)
Base.tan(a::Angle{T}) where T <: Real = tan(a.a)

function regress(X::Matrix{T}, θ::AbstractVector{T2};kwargs...) where T2 <: Angle{T} where T <: Real
    d,n = size(X)
    cy = cos.(θ)
    sy = sin.(θ)
    y = [cy sy]
    ls = LinearRegressionUtils.llsq_stats(permutedims(X), y;kwargs...)
end

function regress(X::Matrix{T}, θ1::AbstractVector{T2}, θ2::AbstractVector{T2};kwargs...) where T2 <: Angle{T} where T <: Real
    d,n = size(X)
    y = [cos.(θ1) sin.(θ1) cos.(θ2) sin.(θ2)]
    ls = LinearRegressionUtils.llsq_stats(permutedims(X),y;kwargs...)
end

function rpca(X::Matrix{T}, θ::AbstractVector{T2}...) where T <: Real where T2
    pca = fit(PCA, X)
    Xp = predict(pca, X)
    ls = regress(Xp, θ...)
    # construct an orthognal basis for β
    w = zeros(T, size(ls.β,1)-1, size(ls.β,2))
    if length(θ) == 1
        q,r = qr(ls.β[1:end-1,:])
        w .= q[:,1:size(ls.β,2)]
    else
        # orthogonalize for each vector
        for i in 1:length(θ)
            jj = ((i-1)*2+1):i*2
            q,r = qr(ls.β[1:end-1,jj])
            w[:,jj] = q[:,jj]
        end
    end

    # project onto the basis
    Z = w'*Xp
    Z, w'*pca.proj'
end

function plot_network_trials(Z::Array{T,3}, θ::Matrix{T};fname::String="network_trials.png", is_saving::Observable{Bool}=Observable(false), kwargs...) where T <: Real
    # slightly hackish
    d = size(Z,1)
    fig = Figure()
    if size(Z,1) == 1
        ax = Axis(fig[1,1])
    else
        ax = Axis3(fig[1,1], xgridvisible=true, ygridvisible=true, zgridvisible=true, viewmode=:stretch)
    end
    ee = dropdims(mean(sum(abs2.(diff(Z,dims=2)), dims=1),dims=3),dims=(1,3))
    cax = Colorbar(fig[1,2], limits=(minimum(θ), maximum(θ)), colormap=:phase)
    cax.label = "θ1"
    _W = diagm(fill(one(T), d))
    W = Observable(_W[1:2,1:d])
    k = Observable(1)
    on(events(fig).keyboardbutton) do event
        if event.action == Keyboard.press || event.action == Keyboard.repeat
            if event.key == Keyboard.r
                q,r = qr(randn(d,d))
                W[] = permutedims(q[:,1:2])
            elseif event.key == Keyboard.c
                k[] = mod(k[],size(θ,2))+1
                cax.label = "θ$(k[])"
            elseif event.key == Keyboard.s
                is_saving[] = true
                save(fname, fig;px_per_unit=8)
                is_saving[] = false
            end

        end
    end
    tl = textlabel!(ax, 0.05, 0.05, text="c : rotate color axis\nr : change projection\ns : save", space=:relative,
              background_color=:black, alpha=0.2, text_align=(:left, :bottom))
    on(is_saving) do _is_saving
        tl.visible[] = !_is_saving
    end
    plot_network_trials!(ax, Z, θ, W;is_saving=is_saving, k=k, kwargs...)
    ax.xlabel = "Time"
    # axis for showing the average speed
    ax2 = Axis(fig[2,1])
    lines!(ax2, 2:(length(ee)+1), ee, color=:black)
    if :trial_events in keys(kwargs)
        ecolors = [:gray, :black, :red, :orange]
        vlines!(ax2, kwargs[:trial_events], color=ecolors, linestyle=:dot)
    end
    ax2.topspinevisible = false
    ax2.rightspinevisible = false
    ax2.xgridvisible = false
    ax2.ygridvisible = false
    ax2.ylabel = "Avg speed"
    ax2.xlabel = "Time"
    rowsize!(fig.layout, 1, Relative(0.8))
    fig
end

function plot_network_trials!(ax, Z::Array{T,3}, θ;kwargs...) where T <: Real
    d = size(Z,1)
    # random projection
    _W = diagm(fill(one(T), d))
    W = Observable(_W[1:2,1:d])
    plot_network_trials!(ax, Z, θ, W;kwargs...)
end

function plot_network_trials!(ax, Z::Array{T,3}, θ::Matrix{T},W::Observable{Matrix{T}};k::Observable{Int64}=Observable(1), trial_events::Vector{Int64}=Int64[], is_saving::Observable{Bool}=Observable(false)) where T <: Real
    _colors = resample_cmap(:phase, size(θ,1))
    sidx = sortperm(θ[:,1])
    vidx = invperm(sidx)
    xt = [1:size(Z,2);]
    μ = mean(Z, dims=(2,3))
    # adjust the limits
    _min, _max = extrema((Z .- μ)[:])
    ylims!(_min, _max)
    if size(Z,1) == 1
        points = [i>size(Z,2) ? Point2f(NaN) : Point2f(xt[i], Z[1,i,j]-μ[1]) for j in 1:size(Z,3) for i in 1:size(Z,2)+1]
    else
        points = lift(W) do _W
            [i>size(Z,2) ? Point3f(NaN) : Point3f(xt[i], (_W*(Z[:,i,j] .-μ[:,1,1]))...) for j in 1:size(Z,3) for i in 1:size(Z,2)+1]
        end
        zlims!(_min, _max)
    end
    colors = lift(k) do _k
        sidx = sortperm(θ[:,_k])
        vidx = invperm(sidx)
        [_colors[vidx[j]] for j in 1:size(Z,3) for i in 1:size(Z,2)+1]
    end
    l = lines!(ax, points, color=colors)
    if !isempty(trial_events)
        #indicate events
        length(trial_events) <= 4 || error("No enough colors for trial_events")
        ecolors = [:gray, :black, :red, :orange]
        points = lift(W) do _W
            [Point3f(_event, _W*(Z[:,_event, j] .- μ[:,1,1])...) for _event in trial_events for j in 1:size(Z,3)]
        end
        colors = [parse(Colorant, ecolors[i]) for i in 1:length(trial_events) for j in 1:size(Z,3)]
        scatter!(ax, points, color=colors)
    end

    ax,l
end

function plot_3d_snapshot(Z::Array{T,3}, θ::Matrix{T};t::Observable{Int64}=Observable(1),show_trajectories::Observable{Bool}=Observable(false), trial_events::Vector{Int64}=Int64[], fname::String="snapshot.png") where T <: Real
    is_saving = Observable(false)
    d,nbins,ntrials = size(Z)
    ee = dropdims(mean(sum(abs2.(diff(Z,dims=2)), dims=1),dims=3),dims=(1,3))
    μ = mean(Z, dims=3)
    # random projection matrix
    _W = diagm(fill(one(T), d))
    W = Observable(_W[1:3,1:d])
    rt = Observable(1)
    do_pause = Observable(true)
    R = 0.01*randn(d,d)
    R  = R - permutedims(R) + diagm(fill(one(T),d))
    on(rt) do _rt
        W[] = W[]*R
    end
    # manually assign colors so that we can use them for the trajectories as well
    k = 1
    acolors = resample_cmap(:phase, size(θ,k))
    sidx = sortperm(θ[:,k])
    vidx = invperm(sidx)
    pcolors = Observable(acolors[vidx])
    points = lift(t,W) do _t, _W
        Point3f.(eachcol(_W*(Z[:,_t, :] .- μ[:,_t,:])))
    end
    traj = lift(t,W) do _t, _W
        [_t >= i >= 1 ? Point3f(_W*(Z[:, i, j] - μ[:,_t])) : Point3f(NaN) for j in 1:size(Z,3) for i in (_t-5):_t+1]
    end
    traj_color = lift(pcolors) do _pc
         [_pc[j] for j in 1:size(θ,1) for i in 1:7]
    end

    # if show trajectories, include fading trajectories of the last 5 points
    fig = Figure()
    ax = Axis3(fig[1,1])
    cax = Colorbar(fig[1,2], limits=(minimum(θ), maximum(θ)), colormap=:phase)
    cax.label = "θ1"
    scatter!(ax, points, color=pcolors)
    ll = lines!(ax, traj, color=traj_color)
    ll.visible = show_trajectories[]
    on(show_trajectories) do _st
        ll.visible = _st
    end
    tt = textlabel!(ax, 0.05, 0.05, text="c : rotate color axis\nr : change projection\np : rotate projection\nt : toggle traces", space=:relative,
              background_color=:black, alpha=0.2, text_align=(:left, :bottom))
    on(events(fig).keyboardbutton) do event
        if event.action == Keyboard.press || event.action == Keyboard.repeat
            if event.key == Keyboard.left
                t[] = max(0, t[]-1)
            elseif event.key == Keyboard.right
                t[] = min(size(Z,2), t[]+1)
            elseif event.key == Keyboard.r
                q,r = qr(randn(d,d))
                W[] = permutedims(q[:,1:3])
            elseif event.key == Keyboard.p
                do_pause[] = !do_pause[]
            elseif event.key == Keyboard.c
                k = mod(k,size(θ,2))+1
                sidx = sortperm(θ[:,k])
                vidx = invperm(sidx)
                pcolors[] = acolors[vidx]
                cax.label = "θ$k"
            elseif event.key == Keyboard.t
                show_trajectories[] = !show_trajectories[]
            elseif event.key == Keyboard.s
                is_saving[] = true
                bn,ex = splitext(fname)
                _fname = replace(fname, ex => "_$(t[])$(ex)")
                save(_fname, fig;px_per_unit=8)
                is_saving[] = false
            end
        end
        #autolimits!(ax)
    end
    # show the average enery
    axe = Axis(fig[2,1])
    lines!(axe, 2:length(ee)+1, ee, color=:black)
    if !isempty(trial_events)
        vlines!(axe, trial_events, color=Cycled(1))
    end
    vlines!(axe, t, color=:black, linestyle=:dot)

    axe.ylabel = "Avg speed"
    axe.xticklabelsvisible = false
    axe.xgridvisible = false
    axe.ygridvisible = false
    axe.topspinevisible = false
    axe.rightspinevisible = false
    rowsize!(fig.layout, 2, Relative(0.2))
    sl = Slider(fig[3,1], range=range(1, stop=size(Z,2), step=1), startvalue=t[], update_while_dragging=true)

    on(t) do _t
        _min,_max = extrema(Z[:,t[], :] .- μ[:,t[],:])
        _mm = maximum(abs.([_min, _max]))
        Δ = 2*_mm
        _min = -_mm - 0.15*Δ
        _max = _mm + 0.15*Δ
        xlims!(ax, _min, _max)
        ylims!(ax, _min, _max)
        zlims!(ax, _min, _max)
    end

    on(is_saving) do _is_saving
        tt.visible[] = !_is_saving
        sl.blockscene.visible[] = !_is_saving
    end

    on(sl.value) do _v
        if t[] != _v
            t[] = _v
        end
    end

    on(t) do _t
        if sl.value[] != _t
            set_close_to!(sl, _t)
        end
    end
    @async while true
        if !do_pause[]
            rt[] = rt[] + 1
        end
        sleep(0.1)
        yield()
    end
    fig
end


end # module MultiDimensionalTimeSeriesPlots
