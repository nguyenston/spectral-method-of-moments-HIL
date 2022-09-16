module OrderRecovery
export reorder_eigenvecs

using LinearAlgebra
using Base.Iterators
using ...Utils

function weighted_gram_and_weights(vec_block)
    norm_block = norm.(eachcol(vec_block))'

    gram_norm = norm_block' * norm_block
    gram_matrix = vec_block' * vec_block

    normalized_gram_matrix = gram_matrix ./ gram_norm
    normalized_gram_matrix[isnan.(normalized_gram_matrix)] .= 1 # correcting invalid entries

    alignment_score_matrix  = abs.(abs.(normalized_gram_matrix) .- 1)
    weighted_score_matrix = alignment_score_matrix .* gram_norm

    return weighted_score_matrix, gram_norm
end

function alignment_mask(alignment_scores, ngroup)
    sorted_alignments = sort(view(alignment_scores, :))
    cutoff_index = div(length(alignment_scores), ngroup)

    tol = (sorted_alignments[cutoff_index] + sorted_alignments[cutoff_index + 1]) / 2
    is_small(x) = abs(x) < tol

    return is_small.(alignment_scores)
end

function generate_groups(group_mask)
    types_of_bitmask = Set{BitVector}()
    for (i, col) in enumerate(eachcol(group_mask))
        push!(types_of_bitmask, col)
    end

    bitmask_to_group(bitmask) = findall(identity, bitmask)
    return bitmask_to_group.(types_of_bitmask)
end

function signed_norm(vec_block)
    vecs = collect(eachcol(vec_block))
    sign_mask = sign.(vecs[1])
    elem_wise_abs(x) = abs.(x)

    unit_vec = normalize(sum(elem_wise_abs.(vecs)) .* sign_mask)
    dot_with_unit_vec(x) = dot(x, unit_vec)

    return dot_with_unit_vec.(vecs)
end

function group_columns_by_profile(indicator_kernel)
    ncol = size(indicator_kernel)[2]
    group_dict = Dict{BitVector, Vector{Int}}()
    for (i, bit_mask) in enumerate(eachcol(indicator_kernel))
        push!(get!(group_dict, bit_mask, []), i)
    end

    profile_assignment = Vector{Int}(undef, ncol)
    profiles = BitVector[]
    cols_with_profile = Vector{Int}[]
    for (i, (prof, group)) in enumerate(pairs(group_dict))
        profile_assignment[group] .= i
        push!(profiles, prof)
        push!(cols_with_profile, group)
    end

    return profile_assignment, profiles, cols_with_profile
end

function find_perm(B, psi)
    (rows, cols) = size(B)
    @assert rows >= cols

    u_normalizer = ((B.^-1) * B') ./ ((psi.^-1) * psi')

    normalized_B = B .* sum(u_normalizer, dims=2)
    map_normalize((a, b)) = (normalize(a), normalize(b))
    splat_dot((a, b)) = dot(a, b)
    dist_to_one(x) = abs(x - 1)

    align_grid = product(eachcol(psi), eachcol(normalized_B))
    raw_perm = dist_to_one.(abs.(splat_dot.(map_normalize.(align_grid))))

    projected_perm = hungarian_algorithm(raw_perm)
end

function reorder_eigenvecs(vecs_input, kernel)
    vecs = deepcopy(vecs_input)
    vecs[abs.(vecs) .< 1e-8 * maximum(vecs)] .= 0

    dim_s = size(kernel)[1]
    dim_os = size(vecs)[1]
    dim_o = div(dim_os, dim_s)
    @assert dim_o * dim_s == dim_os
    
    kernel_inv = inv(kernel)
    kernel_inv[kernel_inv .< 1e-10 * norm(kernel_inv)] .= 0

    partition_grid = dropdims(uniform_partitions(vecs, dim_o, dim_os), dims=2)

    alignment_scores_and_weights = weighted_gram_and_weights.(partition_grid)

    (total_alignment_score, total_weight) = reduce(vectorize(+), alignment_scores_and_weights)
    average_alignment_score = total_alignment_score ./ total_weight
    average_alignment_score[.!isfinite.(average_alignment_score)] .= 1 # correcting invalid cells

    # println("Total alignment score:")
    # pretty_println(total_alignment_score)
    # println("Total weight:")
    # pretty_println(total_weight)
    # println("Average alignment score:")
    # pretty_println(average_alignment_score)

    grouping_mask = alignment_mask(average_alignment_score, dim_o)
    col_groups = generate_groups(grouping_mask)

    # pretty_println(grouping_mask)
    # pretty_println(col_groups)
    @assert length(col_groups) == dim_o

    perm = zeros(Int, dim_o, dim_s)
    for (i, cg) in enumerate(col_groups)
        col_cluster = (x -> x[:, cg]).(partition_grid)
        col_cluster_norm = reduce(hcat, signed_norm.(col_cluster))'

        indicator_kernel = iszero.(kernel_inv)
        (profile_assignments, profiles, kernel_with_prof) = group_columns_by_profile(indicator_kernel)

        weighted_indicator_kernel = indicator_kernel .+ indicator_kernel .* sum(indicator_kernel, dims=1)
        unsigned_col_cluster_norm = col_cluster_norm.^2
        profile_perm = hungarian_algorithm(weighted_indicator_kernel' * unsigned_col_cluster_norm)

        col_with_prof = [Int[] for _ in 1:length(profiles)]
        for (i, pp) in enumerate(profile_perm)
            assigned_profile = profile_assignments[pp]
            push!(col_with_prof[assigned_profile], i)
        end

        cluster_perm = Vector{Int}(undef, dim_s)
        for i in eachindex(profiles)
            valid_rows = .!profiles[i]
            sub_col_cluster = col_cluster_norm[valid_rows, col_with_prof[i]]
            sub_kernel_inv = kernel_inv[valid_rows, kernel_with_prof[i]]

            sub_cluster_perm = find_perm(sub_col_cluster, sub_kernel_inv)
            cluster_perm[col_with_prof[i]] = kernel_with_prof[i][sub_cluster_perm]
        end

        @assert isperm(cluster_perm)

        perm[i, :] = cg[invperm(cluster_perm)]
    end
    perm = cat_cols(perm)
    println("Permutation:")
    println(perm)

    return vecs[:, perm]
end

end # module
