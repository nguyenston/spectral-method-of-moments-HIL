module OptionFramework
export OptFramework, OptionHSM, FaultyActionOptionHSM, step!, Raw3rdOrder, augment_raw_data!, save_moment, load_moment, create_checkpoint!, compare_pilo, compare_pihi

using LinearAlgebra
using StatsBase
using JLD2
using Combinatorics

abstract type OptFramework end

"""
Simulates an agent acting in an environment according to an option framework model
"""
mutable struct OptionHSM <: OptFramework
    o::Int64
    s::Int64
    pilo::Array{Float64, 3}
    phi::Array{Float64, 3}
    pihi::Array{Float64, 3}

    OptionHSM(o, s, pilo, phi, pihi) = new(o, s, pilo, phi, pihi)

    OptionHSM(original::OptionHSM) = begin
        clone = deepcopy(original)
        new(clone.o, clone.s, clone.pilo, clone.phi, clone.pihi)
    end
end

"""
Step the option framework model forward by one time step.
Returns the current (option, state, action)
"""
function step!(model::OptionHSM)
    (dim_o, dim_a, dim_s) = size(model.pilo)
    o = model.o
    s = model.s
    
    distr_a = Weights(model.pilo[o, :, s])
    a = sample(1:dim_a, distr_a)

    distr_s = Weights(model.phi[a, :, s])
    model.s = sample(1:dim_s, distr_s)

    distr_o = Weights(model.pihi[o, :, model.s])
    model.o = sample(1:dim_o, distr_o)

    return [o, s, a]
end


"""
Simulates an agent acting in an environment according to an option framework model
"Faulty action" means sometimes a uniform random action `a_` is taken regardless of the original action `a` chosen by the agent
The history record would still register this as the original action `a` instead of `a_`
"""
mutable struct FaultyActionOptionHSM <: OptFramework
    o::Int64
    s::Int64
    pilo::Array{Float64, 3}
    phi::Array{Float64, 3}
    pihi::Array{Float64, 3}
    fault_rate::Float64

    FaultyActionOptionHSM(o, s, pilo, phi, pihi, fault_rate=0.1) = new(o, s, pilo, phi, pihi, fault_rate)

    FaultyActionOptionHSM(original::FaultyActionOptionHSM) = begin
        clone = deepcopy(original)
        new(clone.o, clone.s, clone.pilo, clone.phi, clone.pihi, clone.fault_rate)
    end
end

"""
Step the option framework model forward by one time step.
Returns the current (option, state, action)
"""
function step!(model::FaultyActionOptionHSM)
    (dim_o, dim_a, dim_s) = size(model.pilo)
    o = model.o
    s = model.s
    
    distr_a = Weights(model.pilo[o, :, s])
    a = sample(1:dim_a, distr_a)
    a_taken = a

    if rand() < model.fault_rate
        a_taken = sample(1:dim_a, uweights(dim_a))
    end

    distr_s = Weights(model.phi[a_taken, :, s])
    model.s = sample(1:dim_s, distr_s)

    distr_o = Weights(model.pihi[o, :, model.s])
    model.o = sample(1:dim_o, distr_o)

    return [o, s, a]
end


"""
A container wrapper around the option framework model
Aside from the model itself also stores the joint probability P(o1, s1, a1, o2, s2, a2, o3, s3, a3)
"""
mutable struct Raw3rdOrder{K <: OptFramework}
    model::K
    buffer::Vector{Vector{Int}}
    data::Array{Float64, 9}
    history::Vector{Tuple{Int, Array{Float64}}}
    T::Int

    sample_path::Vector{Tuple{Int, Int}}
    max_path_length::Int

    Raw3rdOrder(model::K, max_path_length=10000000) where K <: OptFramework = begin
        (dim_o, dim_a, dim_s) = size(model.pilo)
        buffer = [step!(model) for _ in 1:3]
        data = zeros(dim_o, dim_s, dim_a, dim_o, dim_s, dim_a, dim_o, dim_s, dim_a)
        data[vcat(buffer...)...] += 1

        new{typeof(model)}(model, buffer, data, [], 3, (x -> (x[2], x[3])).(buffer), max_path_length)
    end
    
    Raw3rdOrder(original::Raw3rdOrder{K}) where K <: OptFramework = begin
        clone = deepcopy(original)
		modeltype = typeof(clone.model)
        new{K}(K(clone.model), clone.buffer, clone.data, clone.history, clone.T, clone.sample_path, clone.max_path_length)
    end
end

"""
Step the model inside the data wrapper forward and update the running joint probability
""" 
function augment_raw_data!(data)
    popfirst!(data.buffer)
    data_point = step!(data.model)
    push!(data.buffer, data_point)
    data.data[vcat(data.buffer...)...] += 1
    if data.T < data.max_path_length
        push!(data.sample_path, (data_point[2], data_point[3]))
    end
    data.T += 1
end

"""
Create a checkpoint that records the number of samples and the joint probability at that moment
"""
function create_checkpoint!(data)
    push!(data.history, (data.T, deepcopy(data.data)))
end

"""
Save moment data, including the model
"""
function save_moment(name::String, data::Raw3rdOrder)
    jldsave(name; data)
end

"""
Load moment data from a jld2 file
"""
function load_moment(name::String)
    Raw3rdOrder(load(name, "data"))
end

"""
Compute the minimum L2 norm of the difference between truth and prediction
    with respect to the permutations of options
"""
function compare_pilo(est, truth, dim_o)
    (dim_os, _) = size(est)

    compare(perm) = begin
        return norm(est[perm, :] - truth)
    end
    global_perms = [reshape(reshape(1:dim_os, dim_o, :)[p, :], :) for p in permutations(1:dim_o)]
    return minimum(compare.(global_perms))
end

"""
Compute the minimum L2 norm of the difference between truth and prediction
    with respect to the permutations of options
"""
function compare_pihi(est, truth, dim_o)
    (dim_os, _) = size(est)

    compare(perm) = begin
        return norm(est[perm, perm] - truth)
    end
    global_perms = [reshape(reshape(1:dim_os, dim_o, :)[p, :], :) for p in permutations(1:dim_o)]
    return minimum(compare.(global_perms))
end

include("./MomentsMethod/MomentsMethod.jl")
include("./EM.jl")

end # module
