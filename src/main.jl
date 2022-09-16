include("./Utils.jl")
include("./OptionFramework/OptionFramework.jl")
using Random
using .OptionFramework
using .Utils
using LinearAlgebra
using Base.Iterators
using ArgParse
using InteractiveUtils

import .OptionFramework.OrderRecovery.reorder_eigenvecs

function problem_definition()
    dim_s = 4
    dim_a = 2
    dim_o = 2

    fail_rate = 0.1

    pib = [Diagonal([0.8, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.8]) Diagonal([0.2, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.2])]
    pihi_b1 = dense_block_diag([[0.6 0.4; 0.6 0.4], [0.6 0.4; 0.6 0.4], [0.4 0.6; 0.4 0.6], [0.4 0.6; 0.4 0.6]])
    pihi_b0 = (1 - fail_rate) * I(dim_s * dim_o) + (fail_rate / dim_o) * kron(I(dim_s), ones(dim_o, dim_o))

    pihi = extract_block_diag(pib * [pihi_b1; pihi_b0], dim_o, dim_o)
    pilo = [0.7  0.3; 
            0.25 0.75;;; 
            0.8  0.2; 
            0.15 0.85;;; 
            0.65 0.35; 
            0.1  0.9;;; 
            0.9  0.1; 
            0.3  0.7]
    phi = [0.7 0.1 0.1 0.1; 0.25 0.25 0.25 0.25;;; 0.4 0.4 0.1 0.1; 0.1 0.3 0.3 0.3;;; 0.3 0.3 0.3 0.1; 0.1 0.1 0.4 0.4;;; 0.25 0.25 0.25 0.25; 0.1 0.1 0.1 0.7]

    return pihi, pilo, phi
end

function collect_data(modeltype::Type{K}, save_to, load_from="", checkpoints = [1000, 10000, 100000, 1000000, 10000000]) where K <: OptFramework
    (pihi, pilo, phi) = problem_definition()

    if isfile(load_from)
        moment_data::Raw3rdOrder{modeltype} = load_moment(load_from)
    else
        moment_data = Raw3rdOrder(modeltype(1, 1, pilo, phi, pihi))
    end

    for i in moment_data.T+1:maximum(checkpoints)
        augment_raw_data!(moment_data)
        if moment_data.T in checkpoints
            println("saving checkpoints...")
            create_checkpoint!(moment_data)
        end
        if i % 10000 == 0
            save_moment(save_to, moment_data)
        end
    end
end

function generate_moments(raw_data)
    (dim_o, dim_s, dim_a) = size(raw_data)[1:3]

    dim_phi_to_reduce = (1, 4, 6, 7, 8, 9)
    phi = permutedims(dropdims(sum(raw_data, dims=dim_phi_to_reduce), dims=dim_phi_to_reduce), (2, 3, 1))
    for (a, s) in product(1:dim_a, 1:dim_s)
        phi[a, :, s] = normalize(phi[a, :, s], 1)
    end

    moment_sasasa = dropdims(sum(raw_data, dims=(1, 4, 7)), dims=(1, 4, 7))
    normalize!(moment_sasasa, 1)

    moment_ssasa = dropdims(sum(moment_sasasa, dims=2), dims=2)

    perm_col = cat_rows(reshape(1:dim_s * dim_a, dim_s, :))
    moment_ssAsa = [reshape(moment_ssasa[:, :, a, :, :], dim_s^2, :)[:, perm_col] for a in 1:dim_a]

    dim_ssa_to_reduce = (2, 5, 6)
    moment_s2s1a2 = permutedims(dropdims((sum(moment_sasasa, dims=dim_ssa_to_reduce)), dims=dim_ssa_to_reduce), (1, 3, 2))

    (phi = phi, ssAsa = moment_ssAsa, moment_s2s1a2 = moment_s2s1a2)
end

function generate_fudged_moments(phi, ssAsa)
    dim_a = length(ssAsa)
    dim_s = div(size(ssAsa[1])[2], dim_a)
    generate_fudged_moments(phi, ssAsa, I(dim_s))
end

function generate_fudged_moments(phi, ssAsa, kernel)
    dim_a = length(ssAsa)
    dim_s = div(size(ssAsa[1])[2], dim_a)
    min_phi = dropdims(minimum(phi, dims=1), dims=1)'

    raw_fudged_ssAsa = [ssAsa[a] for a in 1:dim_a]
    min_phi = dropdims(minimum(phi, dims=1), dims=1)'
    for a in 1:dim_a
        mask = 1 ./ phi[a, :, :]'
        mask[.!isfinite.(mask)] .= 0
        raw_fudged_ssAsa[a] .*= kron(mask, ones(dim_s, dim_a))
    end

    fudged_ssAsa = [raw_fudged_ssAsa[a] .* kron(kernel, ones(dim_s, dim_a)) for a in 1:dim_a]
    fudged_sssa = sum(fudged_ssAsa)

    (fudged_ssAsa = fudged_ssAsa, fudged_sssa = fudged_sssa)
end

function generate_pilo(matrices, basis, dim_o)
    dim_a = length(matrices)

    basis_inv = inv(basis)
    diagonal_matrices = [basis_inv * m * basis for m in matrices]
    pilo_columns = reduce(hcat, diag.(diagonal_matrices))

    pilo = permutedims(reshape(pilo_columns, dim_o, :, dim_a), (1, 3, 2))

    return pilo
end

function process_and_print(load_from)
    println("Start loading...")
    raw_3rd_order = load_moment(load_from)
    raw_data = raw_3rd_order.history[6][2]
    println("Finished loading.")

    (true_pihi_kron, true_pilo_kron, true_phi_kron) = dense_block_diag.(eachslice.(problem_definition(), dims=3))

    (dim_o, dim_s, dim_a) = size(raw_data)[1:3]
    (phi_estimate, ssAsa, s2s1a2) = generate_moments(raw_data)

    fudge_kernel = ones(dim_s, dim_s) +  I(dim_s)

    fudge_kernel = inv(fudge_kernel)
    (f_ssAsa, f_sssa) = generate_fudged_moments(phi_estimate, ssAsa, fudge_kernel)

    f_sssa_pinv = pinv(f_sssa)
    tensors_o_interest = [f_sssa_pinv * f_ssAsa[a] for a in 1:dim_a]
    eta = randn(dim_a)
    to_be_composed = sum(tensors_o_interest .* eta)
    (_, vecs) = eigen(to_be_composed)

    vecs = reorder_eigenvecs(vecs, fudge_kernel)

    pilo= generate_pilo(tensors_o_interest, vecs, dim_o)
    pilo_kron = dense_block_diag(eachslice(pilo, dims=3))

    s2s1a2_kron = dense_block_diag(eachslice(s2s1a2, dims=3))
    kern_pihi = abs.(pilo_kron * pinv(s2s1a2_kron) * f_sssa * pinv(pilo_kron))

    row_wise_normalize(m, p) = reduce(hcat, normalize.(eachrow(m), p))'

    kern_pihi_seg = uniform_partition_views(kern_pihi, dim_o, dim_o)
    weighted_kern_pihi_seg = kern_pihi_seg .* abs.(fudge_kernel)
    smeared_pihi = dropdims(sum(weighted_kern_pihi_seg, dims=1), dims=1)
    pihi_kron = row_wise_normalize(dense_block_diag(smeared_pihi), 1)

    println("Fudge kernel:")
    pretty_println(fudge_kernel)
    pretty_println(inv(fudge_kernel))
    println("PILO comparison:")
    pretty_println(pilo_kron)
    pretty_println(true_pilo_kron)
    println("PIHI comparison:")
    pretty_println(pihi_kron)
    pretty_println(true_pihi_kron)
end

function main()
    settings = ArgParseSettings(description = "Spectral method for HIL")
    @add_arg_table(settings, begin
        "collect-data", "c"
            help = "collect data and save to file"
            action = :command
        "process-data", "p"
            help = "process collected data and print result"
            action = :command
    end)
    @add_arg_table(settings["collect-data"], begin
        "save-loc"
            help = "location to save data"
            required = true
            metavar = "SAVE_LOC"
        "--load-from", "-l"
            help = "load location to continue collecting data"
            default = ""
        "--model-type", "-m"
            help = "type of model to run experiment on"
            default = "OptionHSM"
    end)
    @add_arg_table(settings["process-data"], begin
        "load-loc"
            help = "load location to continue collecting data"
            required = true
            metavar = "LOAD_LOC"
    end)
    parsed_args = parse_args(ARGS, settings)
    if parsed_args["%COMMAND%"] == "collect-data"
        args = parsed_args["collect-data"]

        modeltype_string = string.(subtypes(OptFramework))
        modeltype_datatype = subtypes(OptFramework)
        modeltype_table = Dict(Pair(x...) for x in zip(modeltype_string, modeltype_datatype))

        checkpoints = [1000, 10000, 100000, 1000000, 10000000, 100000000]
        modeltype = modeltype_table[args["model-type"]]
        save_to = args["save-loc"]
        load_from = args["load-from"]
        collect_data(modeltype, save_to, load_from, checkpoints)
    elseif parsed_args["%COMMAND%"] == "process-data"
        args = parsed_args["process-data"]
        load_from = args["load-loc"]

        process_and_print(load_from)
    end

end

main()
