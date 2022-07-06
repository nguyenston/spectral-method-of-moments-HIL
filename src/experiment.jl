include("./OptionFramework.jl")
include("./Utils.jl")
using .OptionFramework
using .Utils
using LinearAlgebra
using Base.Iterators

function collect_data()
    dim_s = 4
    dim_a = 2
    dim_o = 2

    fail_rate = 0.1

    pib = [Diagonal([0.8, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.8]) Diagonal([0.2, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.2])]
    pihi_b1 = dense_block_diag([[0.6 0.4; 0.6 0.4], [0.6 0.4; 0.6 0.4], [0.4 0.6; 0.4 0.6], [0.4 0.6; 0.4 0.6]])
    pihi_b0 = (1 - fail_rate) * I(dim_s * dim_o) + (fail_rate / dim_o) * kron(I(dim_s), ones(dim_o, dim_o))

    pihi = extract_block_diag(pib * [pihi_b1; pihi_b0], dim_o, dim_o, dim_s)
    pilo = extract_block_diag(kron(I(dim_s), [0.7 0.3; 0.3 0.7]), dim_o, dim_a, dim_s)
    phi = [1 0 0 0; 0.25 0.25 0.25 0.25;;; 0.5 0.5 0 0; 0 1/3 1/3 1/3;;; 1/3 1/3 1/3 0; 0 0 0.5 0.5;;; 0.25 0.25 0.25 0.25; 0 0 0 1]

    if isfile("save_data.jld2")
        moment_data = load_moment("save_data.jld2")
    else
        moment_data = Raw3rdOrder(OptionHSM(1, 1, pilo, phi, pihi))
    end

    checkpoints = [1000, 100000, 10000000]

    for i in 4:10000000
        augment_raw_data!(moment_data)
        if moment_data.T in checkpoints
            create_checkpoint!(moment_data)
        end
        if i % 10000 == 0
            save_moment("save_data.jld2", moment_data)
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
    moment_sssa = sum(moment_ssAsa)

    (phi = phi, ssAsa = moment_ssAsa, sssa = moment_sssa)
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

function main()
    raw_3rd_order = load_moment("save_data.jld2")
    raw_data = raw_3rd_order.data

    (dim_o, dim_s, dim_a) = size(raw_data)[1:3]
    (phi, ssAsa, sssa) = generate_moments(raw_data)
    (f_ssAsa, f_sssa) = generate_fudged_moments(phi, ssAsa)

    tensors_o_interest = [pinv(f_sssa) * f_ssAsa[a] for a in 1:dim_a]
    eta = randn(dim_a)
    to_be_composed = sum(tensors_o_interest .* eta)
    (vals, vecs) = eigen(to_be_composed)
    vecs[vecs .< 1e-10] .= 0

    show(stdout, "text/plain", vals)
    println()
    show(stdout, "text/plain", vecs)
    println()

    fail_rate = 0.1
    pib = [Diagonal([0.8, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.8]) Diagonal([0.2, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.2])]
    pihi_b1 = dense_block_diag([[0.6 0.4; 0.6 0.4], [0.6 0.4; 0.6 0.4], [0.4 0.6; 0.4 0.6], [0.4 0.6; 0.4 0.6]])
    pihi_b0 = (1 - fail_rate) * I(dim_s * dim_o) + (fail_rate / dim_o) * kron(I(dim_s), ones(dim_o, dim_o))
    pihi_kron = pib * [pihi_b1; pihi_b0]

    pilo_kron = kron(I(dim_s), [0.7 0.3; 0.3 0.7])
    show(stdout, "text/plain", pilo_kron)
    println()
    show(stdout, "text/plain", pihi_kron * pilo_kron)
    println()
end

main()