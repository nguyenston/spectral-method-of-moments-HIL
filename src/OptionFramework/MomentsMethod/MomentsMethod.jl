module MomentsMethod
export generate_moments, generate_fudged_moments, generate_pilo, collect_data, process_and_print

include("./OrderRecovery.jl")

using LinearAlgebra
using Base.Iterators
using ...Utils
using ..OptionFramework
using .OrderRecovery


"""
Compute various moments from the statistics of state-action pairs across 3 time step
phi[a1, s2, s1] = P(s2 | s1, a1)
ssAsa[a][(s2 - 1) * dim_s + s1, (s3 - 1) * dim_a + a3] = P(s1, s2, a2 = a, s3, a3)
s2s1a2[s1, a2, s2] = P(s1, s2, a2)
"""
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

    (phi = phi, ssAsa = moment_ssAsa, s2s1a2 = moment_s2s1a2)
end

"""
Compute the moment surogate through normalization and apply fudge kernel
If no argument is passed for kernel, Identity is assumed
"""
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

"""
recover the low-level policy from the diagonalizable complex and its eigen basis
"""
function generate_pilo(matrices, basis, dim_o)
    dim_a = length(matrices)

    basis_inv = inv(basis)
    diagonal_matrices = [basis_inv * m * basis for m in matrices]
    pilo_columns = reduce(hcat, diag.(diagonal_matrices))

    pilo = permutedims(reshape(pilo_columns, dim_o, :, dim_a), (1, 3, 2))

    return pilo
end

function collect_data(modeltype::Type{K}, problem, save_to, load_from="", checkpoints = [1000, 10000, 100000, 1000000, 10000000]) where K <: OptFramework
    (pihi, pilo, phi) = problem()
    max_T = maximum(checkpoints)

    if isfile(load_from)
        moment_data::Raw3rdOrder{modeltype} = load_moment(load_from)
    else
        moment_data = Raw3rdOrder(modeltype(1, 1, pilo, phi, pihi))
    end

    if isfile(save_to)
        print("This file already exists, overwrite? (y/n) ")
        overwrite = readline()
        if overwrite != "y"
            return
        end
    end

    for i in moment_data.T+1:max_T
        augment_raw_data!(moment_data)
        if moment_data.T in checkpoints
            println("saving checkpoints at ", i)
            create_checkpoint!(moment_data)
        end
        if i % 1000 == 0
            print("Sample: ", i, "/", max_T, "\r")
        end
        if i % 5000000 == 0 || i == max_T
            save_moment(save_to, moment_data)
        end
    end
end

function process_and_print(problem, raw_3rd_order, index = 0; terse = false)
    if index == 0
        index = length(raw_3rd_order.history)
    end
    raw_data = raw_3rd_order.history[index][2]

    (true_pihi_kron, true_pilo_kron, true_phi_kron) = dense_block_diag.(eachslice.(problem(), dims=3))

    (dim_o, dim_s, dim_a) = size(raw_data)[1:3]
    (phi_estimate, ssAsa, s2s1a2) = generate_moments(raw_data)

    fudge_kernel = ones(dim_s, dim_s) +  2I(dim_s)

    fudge_kernel = inv(fudge_kernel)
    (f_ssAsa, f_sssa) = generate_fudged_moments(phi_estimate, ssAsa, fudge_kernel)

    f_sssa_pinv = pinv(f_sssa)
    tensors_o_interest = [f_sssa_pinv * f_ssAsa[a] for a in 1:dim_a]
    eta = randn(dim_a)
    to_be_composed = sum(tensors_o_interest .* eta)
    (_, vecs) = eigen(to_be_composed)

    vecs = reorder_eigenvecs(vecs, fudge_kernel; terse = terse)

    rotate_to_real = x -> abs(x) * sign(real(x))
    pilo= rotate_to_real.(generate_pilo(tensors_o_interest, vecs, dim_o))
    pilo_kron = dense_block_diag(eachslice(pilo, dims=3))

    s2s1a2_kron = dense_block_diag(eachslice(s2s1a2, dims=3))
    kern_pihi = abs.(pilo_kron * pinv(s2s1a2_kron) * f_sssa * pinv(pilo_kron))

    row_wise_normalize(m, p) = reduce(hcat, normalize.(eachrow(m), p))'

    kern_pihi_seg = uniform_partition_views(kern_pihi, dim_o, dim_o)
    weighted_kern_pihi_seg = kern_pihi_seg .* abs.(fudge_kernel)
    smeared_pihi = dropdims(sum(weighted_kern_pihi_seg, dims=1), dims=1)
    pihi_kron = row_wise_normalize(dense_block_diag(smeared_pihi), 1)

    if terse
        T = raw_3rd_order.history[index][1]
        err_lo = compare_pilo(pilo_kron, true_pilo_kron, dim_o)
        err_hi = compare_pihi(pihi_kron, true_pihi_kron, dim_o)
        println(T, " ",  sqrt(err_lo^2 + err_hi^2))
    else
        println("Fudge kernel:")
        pretty_println(fudge_kernel)
        pretty_println(inv(fudge_kernel))
        println("PILO comparison:")
        pretty_println(pilo_kron)
        pretty_println(true_pilo_kron)
        println("PIHI comparison:")
        pretty_println(pihi_kron)
        pretty_println(true_pihi_kron)

        println("PILO diff:")
        println(compare_pilo(pilo_kron, true_pilo_kron, dim_o))
        println("PIHI diff:")
        println(compare_pihi(pihi_kron, true_pihi_kron, dim_o))
    end
end
end
