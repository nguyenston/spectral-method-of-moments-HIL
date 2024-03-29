include("./Utils.jl")
include("./OptionFramework/OptionFramework.jl")
using Random
using .OptionFramework
using .OptionFramework.MomentsMethod
using .OptionFramework.EM
using .Utils
using LinearAlgebra
using ArgParse
using InteractiveUtils
using LinearAlgebra
using Base.Iterators
using JLD2

function problem_definition()
    dim_s = 4
    dim_a = 2
    dim_o = 2

    fail_rate = 0.1

    pib = [Diagonal([0.8, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.8]) Diagonal([0.2, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.2])]
    pihi_b1 = dense_block_diag([[0.6 0.4; 0.6 0.4], [0.6 0.4; 0.6 0.4], [0.4 0.6; 0.4 0.6], [0.4 0.6; 0.4 0.6]])
    pihi_b0 = (1 - fail_rate) * I(dim_s * dim_o) + (fail_rate / dim_o) * kron(I(dim_s), ones(dim_o, dim_o))

    pihi = extract_block_diag(pib * [pihi_b1; pihi_b0], dim_o, dim_o)
    pilo = [0.6  0.4; 
            0.1  0.9;;; 
            0.7  0.3; 
            0.15 0.85;;; 
            0.8  0.2; 
            0.3  0.7;;; 
            0.9  0.1; 
            0.35 0.65]
    phi = [0.7 0.1 0.1 0.1; 0.25 0.25 0.25 0.25;;; 0.4 0.4 0.1 0.1; 0.1 0.3 0.3 0.3;;; 0.3 0.3 0.3 0.1; 0.1 0.1 0.4 0.4;;; 0.25 0.25 0.25 0.25; 0.1 0.1 0.1 0.7]

    return pihi, pilo, phi
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
        "EM"
            help = "process collected data by the EM method"
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
        "--checkpoint-index", "-i"
            help = "choose a checkpoint"
            arg_type = Int
            default = 0
        "--terse", "-t"
            help = "only print the final difference from ground truth"
            action = :store_true
    end)
    @add_arg_table(settings["EM"], begin
        "load-loc"
            help = "load location to continue collecting data"
            required = true
            metavar = "LOAD_LOC"
        "--n-sample", "-n"
            help = "specify the number of sample to process"
            arg_type = Int
            default = 0
    end)
    parsed_args = parse_args(ARGS, settings)
    if parsed_args["%COMMAND%"] == "collect-data"
        args = parsed_args["collect-data"]

        modeltype_string = string.(subtypes(OptFramework))
        modeltype_datatype = subtypes(OptFramework)
        modeltype_table = Dict(Pair(x...) for x in zip(modeltype_string, modeltype_datatype))

        checkpoints = [2^i for i in 10:30]
        modeltype = modeltype_table[args["model-type"]]
        save_to = args["save-loc"]
        load_from = args["load-from"]
        collect_data(modeltype, problem_definition, save_to, load_from, checkpoints)
    elseif parsed_args["%COMMAND%"] == "process-data"
        args = parsed_args["process-data"]
        load_from = args["load-loc"]
        terse = args["terse"]

        if !terse
            println("Start loading...")
            raw_3rd_order = load_moment(load_from)
            println("Finished loading.")
        else
            raw_3rd_order = load_moment(load_from)
        end

        index = args["checkpoint-index"]
        if index == 0
            for (i, history) in enumerate(raw_3rd_order.history)
                println(i, ") ", history[1])
            end
            print("Please choose the number of sample (1-", length(raw_3rd_order.history), "): ")
            index = parse(Int, readline())
        end
        process_and_print(problem_definition, raw_3rd_order, index; terse = terse)
    elseif parsed_args["%COMMAND%"] == "EM"
        args = parsed_args["EM"]
        load_from = args["load-loc"]
        n_sample = args["n-sample"]

        # println("Start loading...")
        raw_3rd_order = load_moment(load_from)
        # println("Finished loading.")
        #
        # println("Sample path length:")
        # println(length(raw_3rd_order.sample_path))
        if n_sample == 0
            n_sample = length(raw_3rd_order.sample_path)
        end
        # pihi = [0.1  0.9; 
        #         0.3  0.7;;; 
        #         0.5  0.5; 
        #         0.4 0.6;;; 
        #         0.6  0.4; 
        #         0.5  0.5;;; 
        #         0.2  0.8; 
        #         0.65 0.35]
        # pilo = [0.3  0.7; 
        #         0.3  0.7;;; 
        #         0.9  0.1; 
        #         0.35 0.65;;; 
        #         0.6  0.4; 
        #         0.1  0.9;;; 
        #         0.2  0.8; 
        #         0.65 0.35]

        if isfile("./em_seed.jld2")
            pilo, pihi = load("./em_seed.jld2", "policies")
        else
            row_wise_normalize(m, p) = reduce(hcat, normalize.(eachrow(m), p))'
            # (pihi_kron, pilo_kron, _) = dense_block_diag.(eachslice.(problem_definition(), dims=3))
            pihi_kron = row_wise_normalize(dense_block_diag(eachslice(abs.(rand(2, 2, 4).^3), dims=3)), 1)
            pilo_kron = row_wise_normalize(dense_block_diag(eachslice(abs.(rand(2, 2, 4).^3), dims=3)), 1)
            pihi = extract_block_diag(pihi_kron, 2, 2)
            pilo = extract_block_diag(pilo_kron, 2, 2)
            policies = (pilo, pihi)
            jldsave("./em_seed_local_optima.jld2"; policies)
        end

        # pilo = 0.5 * ones(2, 2, 4)
        # pihi = 0.5 * ones(2, 2, 4)
        pilo_kron = dense_block_diag(eachslice(pilo, dims=3))
        pihi_kron = dense_block_diag(eachslice(pihi, dims=3))

        (pilo, pihi) = iterate_and_print(problem_definition, raw_3rd_order, n_sample,pilo, pihi, T=000)

        # new_pilo_kron = dense_block_diag(eachslice(pilo, dims=3))
        # new_pihi_kron = dense_block_diag(eachslice(pihi, dims=3))
        # (true_pihi_kron, true_pilo_kron, _) = dense_block_diag.(eachslice.(problem_definition(), dims=3))
        pretty_println(pilo_kron)
        # pretty_println(new_pilo_kron)
        # pretty_println(true_pilo_kron)
        pretty_println(pihi_kron)
        # pretty_println(new_pihi_kron)
        # pretty_println(true_pihi_kron)
    end

end

main()
