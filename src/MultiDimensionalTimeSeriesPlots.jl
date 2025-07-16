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
    fig = Figure()
    if size(Z,1) == 1
        ax = Axis(fig[1,1])
    else
        ax = Axis3(fig[1,1], xgridvisible=true, ygridvisible=true, zgridvisible=true, viewmode=:stretch)
    end
    plot_network_trials!(ax, Z, θ;kwargs...)
    ax.xlabel = "Time"
    fig
end

function plot_network_trials!(ax, Z::Array{T,3}, θ::Vector{T};cidx::Observable{Int64}=Observable(1), trial_events::Vector{Int64}=Int64[]) where T <: Real
    _colors = resample_cmap(:phase, length(θ))
    sidx = sortperm(θ)
    vidx = invperm(sidx)
    xt = [1:size(Z,2);]
    # rotate around the z-axes
    R = [1.0f0 0.0f0 0.0f0
         0.0f0 1.0f0 -0.01f0;
         0.0f0 0.01f0 1.0f0]

    μ = mean(Z, dims=(2,3))
    # adjust the limits
    _min, _max = extrema((Z .- μ)[:])    
    ylims!(_min, _max)
    if size(Z,1) == 1
        points = [i>size(Z,2) ? Point2f(NaN) : Point2f(xt[i], Z[1,i,j]-μ[1]) for j in 1:size(Z,3) for i in 1:size(Z,2)+1]
    else
        points = [i>size(Z,2) ? Point3f(NaN) : Point3f(xt[i], Z[1,i,j]-μ[1], Z[2,i,j]-μ[2]) for j in 1:size(Z,3) for i in 1:size(Z,2)+1]
        zlims!(_min, _max)
    end
    colors = [_colors[vidx[j]] for j in 1:size(Z,3) for i in 1:size(Z,2)+1]
    l = lines!(ax, points, color=colors)
    if !isempty(trial_events)
        #indicate events
        ecolors = [:gray, :black, :red]
        points = [Point3f(_event, Z[1,_event, j]-μ[1], Z[2, _event,j]-μ[2]) for _event in trial_events for j in 1:size(Z,3)] 
        colors = [parse(Colorant, ec) for ec in ecolors for j in 1:size(Z,3)]
        scatter!(ax, points, color=colors)
    end

    ax,l
end

function plot_3d_snapshot(Z::Array{T,3}, θ::Vector{T};t::Observable{Int64}=Observable(1),show_trajectories=false) where T <: Real
    d,nbins,ntrials = size(Z)
    # manually assign colors so that we can use them for the trajectories as well
    acolors = resample_cmap(:phase, length(θ))
    sidx = sortperm(θ)
    vidx = invperm(sidx)
    points = lift(t) do _t
        Point3f.(eachcol(Z[:,_t, :]))
    end
    traj = lift(t) do _t
        [_t >= i >= 1 ? Point3f(Z[1:3, i, j]) : Point3f(NaN) for j in 1:size(Z,3) for i in (_t-5):_t+1]
    end
    traj_color = [acolors[vidx[j]] for j in 1:length(θ) for i in 1:7] 
    @show length(traj_color) length(traj[])

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
