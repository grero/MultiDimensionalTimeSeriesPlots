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

function regress(X::Matrix{T}, θ::Vector{T2};kwargs...) where T2 <: Angle{T} where T <: Real
    d,n = size(X)
    cy = cos.(θ)
    sy = sin.(θ)
    y = [cy sy]
    ls = LinearRegressionUtils.llsq_stats(permutedims(X), y;kwargs...)
end

function regress(X::Matrix{T}, θ1::Vector{T2}, θ2::Vector{T2};kwargs...) where T2 <: Angle{T} where T <: Real
    d,n = size(X)
    y = [cos.(θ1) sin.(θ1) cos.(θ2) sin.(θ2)]
    ls = LinearRegressionUtils.llsq_stats(permutedims(X),y;kwargs...)
end

function rpca(X::Matrix{T}, θ::Vector{T2}...) where T <: Real where T2
    pca = fit(PCA, X)
    Xp = predict(pca, X)
    ls = regress(Xp, θ...)
    # construct an orthognal basis for β
    q,r = qr(ls.β[1:end-1,:])
    w = q[:,1:size(ls.β,2)]

    # project onto the basis
    Z = w'*Xp
    Z, w'*pca.proj'
end

function plot_network_trials(Z::Array{T,3}, θ::Vector{T};kwargs...) where T <: Real
    # slightly hackish
    d = size(Z,1)
    fig = Figure()
    if size(Z,1) == 1
        ax = Axis(fig[1,1])
    else
        ax = Axis3(fig[1,1], xgridvisible=true, ygridvisible=true, zgridvisible=true, viewmode=:stretch)
    end
    _W = diagm(fill(one(T), d))
    W = Observable(_W[1:2,1:d])
    on(events(fig).keyboardbutton) do event
        if event.action == Keyboard.press || event.action == Keyboard.repeat
            if event.key == Keyboard.r
                q,r = qr(randn(d,d))
                W[] = permutedims(q[:,1:2])
            end
        end
    end
    plot_network_trials!(ax, Z, θ, W;kwargs...)
    ax.xlabel = "Time"
    fig
end

function plot_network_trials!(ax, Z::Array{T,3}, θ::Vector{T};kwargs...) where T <: Real
    d = size(Z,1)
    # random projection
    _W = diagm(fill(one(T), d))
    W = Observable(_W[1:2,1:d])
    plot_network_trials!(ax, Z, θ, W;kwargs...)
end

function plot_network_trials!(ax, Z::Array{T,3}, θ::Vector{T},W::Observable{Matrix{T}};trial_events::Vector{Int64}=Int64[]) where T <: Real
    _colors = resample_cmap(:phase, length(θ))
    sidx = sortperm(θ)
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
    colors = [_colors[vidx[j]] for j in 1:size(Z,3) for i in 1:size(Z,2)+1]
    l = lines!(ax, points, color=colors)
    if !isempty(trial_events)
        #indicate events
        ecolors = [:gray, :black, :red]
        points = lift(W) do _W
            [Point3f(_event, _W*(Z[:,_event, j] .- μ[:,1,1])...) for _event in trial_events for j in 1:size(Z,3)] 
        end
        colors = [parse(Colorant, ec) for ec in ecolors for j in 1:size(Z,3)]
        scatter!(ax, points, color=colors)
    end

    ax,l
end

function plot_3d_snapshot(Z::Array{T,3}, θ::Vector{T};t::Observable{Int64}=Observable(1),show_trajectories=false) where T <: Real
    d,nbins,ntrials = size(Z)
    # random projection matrix
    _W = diagm(fill(one(T), d))
    W = Observable(_W[1:3,1:d])
    # manually assign colors so that we can use them for the trajectories as well
    acolors = resample_cmap(:phase, length(θ))
    sidx = sortperm(θ)
    vidx = invperm(sidx)
    points = lift(t,W) do _t, _W
        Point3f.(eachcol(_W*Z[:,_t, :]))
    end
    traj = lift(t,W) do _t, _W
        [_t >= i >= 1 ? Point3f(_W*Z[:, i, j]) : Point3f(NaN) for j in 1:size(Z,3) for i in (_t-5):_t+1]
    end
    traj_color = [acolors[vidx[j]] for j in 1:length(θ) for i in 1:7] 

    # if show trajectories, include fading trajectories of the last 5 points
    fig = Figure()
    ax = Axis3(fig[1,1])
    scatter!(ax, points, color=acolors[vidx])
    if show_trajectories
        lines!(ax, traj, color=traj_color)
    end
    on(events(fig).keyboardbutton) do event
        if event.action == Keyboard.press || event.action == Keyboard.repeat
            if event.key == Keyboard.left
                t[] = max(0, t[]-1)
            elseif event.key == Keyboard.right
                t[] = min(size(Z,2), t[]+1)
            elseif event.key == Keyboard.r
                q,r = qr(randn(d,d))
                W[] = permutedims(q[:,1:3])
            end
        end
        autolimits!(ax)
    end
    sl = Slider(fig[2,1], range=range(1, stop=size(Z,2), step=1), startvalue=t[], update_while_dragging=true) 
    on(sl.value) do _v
        if t[] != _v
            t[] = _v
            autolimits!(ax)
        end
    end

    on(t) do _t
        if sl.value[] != _t
            set_close_to!(sl, _t)
        end
    end
    fig
end


end # module MultiDimensionalTimeSeriesPlots
