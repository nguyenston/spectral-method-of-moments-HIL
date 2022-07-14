module OrderRecovery
export reorder_eigenvecs

using LinearAlgebra
using Base.Iterators
using ...Utils

function generate_row_order(kernel_inv)
    is_overlap(bitstr1, bitstr2) = begin
        reduce(|, bitstr1 .& bitstr2)
    end

    num_rows = size(kernel_inv)[1]
    profile = abs.(kernel_inv) .> 0
    order = [1]
    
    running_union = profile[1, :]
    remaining_index = collect(2:num_rows)
    remaining = collect(zip(remaining_index, [profile[i, :] for i in remaining_index]))
    connected_rows = findall(x->is_overlap(x[2], running_union), remaining)
    
    for _ in 1:num_rows
        cr_index = [remaining[cr][1] for cr in connected_rows]
        for cr in cr_index
            running_union = running_union .|| profile[cr, :]
        end
        
        append!(order, cr_index)
        remaining_index = setdiff(remaining_index, cr_index)
        remaining = collect(zip(remaining_index, [profile[i, :] for i in remaining_index]))

        connected_rows = findall(x->is_overlap(x[2], running_union), remaining)
        
        if isempty(connected_rows)
            break
        end
    end

    return order
end

function segment_wise_norm(vecs, segments)
    # reduce(vcat, [[norm(col) for col in eachcol(vecs[s[1]:s[2], :])]' for s in segments])
    result = []
    for s in segments
        seg = vecs[s[1]:s[2], :]
        signs = sign.(seg[:, 1])
        regular_sign_seg = signs .* abs.(seg)
        normed_vec = sum(normalize.(eachcol(regular_sign_seg))) / size(seg)[2]
        push!(result, normed_vec' * seg)
    end
    return reduce(vcat, result)
end

function linearly_dependent_groups(ivs, dim_o)  
    group_index = Set{Set{Int}}()

    indices = [iv[1] for iv in ivs]
    vectors = normalize.([iv[2] for iv in ivs])

    alignment(x) = abs(abs(x[1]' * x[2]) - 1)

    alignment_grid = alignment.(product(vectors, vectors))
    show(IOContext(stdout, :limit=>false), "text/plain", reduce(hcat, vectors))
    println()
    show(IOContext(stdout, :limit=>false), "text/plain", alignment_grid)
    println()

    sorted_alignments = sort(alignment_grid[:])
    cutoff = div(length(alignment_grid), dim_o)
    tol = (sorted_alignments[cutoff] + sorted_alignments[cutoff + 1]) / 2
    println(tol)

    is_small(tol) = x -> x <= tol
    for col in eachcol(alignment_grid)
        push!(group_index, Set(indices[findall(is_small(tol), col)]))
    end


    filter_small(x, tol) = x > tol ? x : 0
    filtered_grid = filter_small.(alignment_grid, tol)
    show(IOContext(stdout, :limit=>false), "text/plain", filtered_grid)
    println()

    @assert length(group_index) == dim_o
    return group_index
end

function linearly_dependent_groups(ivs, dim_o, group_index_input)
    group_index = Set{Set{Int}}()
    indices = [iv[1] for iv in ivs]
    vectors = normalize.([iv[2] for iv in ivs])

    preassigned_indices = []
    for i in 1:length(indices)
        for group in group_index_input
            if indices[i] in group
                push!(preassigned_indices, (i, group))
                break
            end
        end
    end
    

    alignment(x) = abs(abs(x[1]' * x[2]) - 1)

    alignment_grid = alignment.(product(vectors, vectors))
    show(IOContext(stdout, :limit=>false), "text/plain", reduce(hcat, vectors))
    println()
    show(IOContext(stdout, :limit=>false), "text/plain", alignment_grid)
    println()

    sorted_alignments = sort(alignment_grid[:])
    cutoff = div(length(alignment_grid), dim_o)
    tol = (sorted_alignments[cutoff] + sorted_alignments[cutoff + 1]) / 2
    println(tol)

    is_small(tol) = x -> x <= tol
    for (i, g) in preassigned_indices
        push!(group_index, union(g, Set(indices[findall(is_small(tol), alignment_grid[:, i])])))
    end


    filter_small(x, tol) = x > tol ? x : 0
    filtered_grid = filter_small.(alignment_grid, tol)
    show(IOContext(stdout, :limit=>false), "text/plain", filtered_grid)
    println()

    @assert length(group_index) == dim_o
    return group_index
end

function vector_profile(vec, tol=1e-7)
    return abs.(vec) .> tol * norm(vec)
end

function group_by_profile(ivs)
    group_profile = []
    group_index = []
    
    for (index, vector) in ivs
        profile = vector_profile(vector)
        spawn_new = true
        
        for j in 1:length(group_profile)
            if profile == group_profile[j]
                push!(group_index[j], index)
                spawn_new = false
                break
            end
        end

        if spawn_new
            push!(group_profile, profile)
            push!(group_index, [index])         
        end
    end

    return group_profile, group_index
end

function classify_by_profile(ivs, profiles, tol=1e-7)
    group = [[] for _ in 1:length(profiles)]
    indices = [iv[1] for iv in ivs]
    vectors = reduce(hcat, [iv[2] for iv in ivs])
    vector_profiles = reduce(vcat, transpose.(vector_profile.(eachrow(vectors), tol)))
    for (i, index) in enumerate(indices)
        profile = vector_profiles[:, i]
        println(profile)
        found = false

        for j in 1:length(profiles)
            if profile == profiles[j]
                push!(group[j], index)
                found = true
                break
            end
        end

        if !found
            error("no matching profile")
        end
    end
    return group
end

function intersect_with_tolerance(
    set1::AbstractVector{T},
    set2::AbstractVector{K},
    set1_weight::Real=1;
    rtol=max(eps(T), eps(K))^0.25
) where {T, K <: Number}

    grid = collect(product(set1, set2))
    bit_mask = (x->isapprox(x[1], x[2], rtol=rtol)).(grid)

    @assert reduce(&, sum(bit_mask, dims=1) .<= 1) "Non_unique matches found"

    index_grid = collect(product(1:length(set1), 1:length(set2)))
    matched_indices = index_grid[bit_mask]
    m1 = [match[1] for match in matched_indices]
    m2 = [match[2] for match in matched_indices]
    matched_avg_values = (set1_weight * set1[m1] + set2[m2]) / (set1_weight + 1)
    
    return matched_avg_values, m1, m2
end

function match_making(cluster_group, kernel_group, ratio_matrix, rtol=eps()^0.25)
    dim_s = size(ratio_matrix)[1]
    reordered_column = fill(0, dim_s)
    for j in 1:length(kernel_group)
        c_group = cluster_group[j]
        k_group = kernel_group[j]
        println(c_group, k_group)
        @assert length(c_group) == length(k_group)
        
        r_mat = ratio_matrix[k_group, c_group]

        running_avg = r_mat[:, 1]
        matched_indices = hcat(1:length(c_group))
        for i in 2:length(c_group)
            (matched_avg_values, m1, m2) = intersect_with_tolerance(running_avg, r_mat[:, i], i-1, rtol=rtol)
            
            running_avg = matched_avg_values
            matched_indices = hcat(matched_indices[m1, :], m2)
        end
        
        show(stdout, "text/plain", matched_indices)
        println()
        @assert size(matched_indices) == (1, length(c_group))
        for (m, n) in enumerate(matched_indices)
            kernel_index = k_group[n]
            @assert reordered_column[kernel_index] == 0
            
            reordered_column[kernel_index] = c_group[m]
        end
    end
    return reordered_column
end

function reorder_eigenvecs(vecs, kernel; rtol=eps()^0.25, ptol=1e-1)
    dim_s = size(kernel)[1]
    dim_os = size(vecs)[1]
    dim_o = div(dim_os, dim_s)
    @assert dim_o * dim_s == dim_os
    
    kernel_inv = inv(kernel)
    ranges = collect(zip(1:dim_o:dim_os, dim_o:dim_o:dim_os))

    order = generate_row_order(kernel_inv)

    @assert length(order) == dim_s

    sub_rows = vecs[ranges[order[1]][1]:ranges[order[1]][2], :]
    numbered_col = enumerate(eachcol(sub_rows))
    non_zero_col = collect(Iterators.filter(x -> !iszero(x[2]), numbered_col))
    sorted_col = linearly_dependent_groups(non_zero_col, dim_o)

    @assert length(sorted_col) == dim_o

    for range in ranges[order[2:end]]
        sub_rows = vecs[range[1]:range[2], :]
        numbered_col = enumerate(eachcol(sub_rows))
        non_zero_col = collect(Iterators.filter(x -> !iszero(x[2]), numbered_col))
        sorted_col = linearly_dependent_groups(non_zero_col, dim_o, sorted_col)
    end

    sorted_col = collect.(sorted_col)
    (kernel_profile, kernel_group) = group_by_profile(enumerate(eachrow(kernel_inv)))
    show(stdout, "text/plain", sorted_col)
    println()
    show(stdout, "text/plain", kernel_inv)
    println()

    
    first_non_zero_rows = [findfirst(x->abs(x)>0, col) for col in eachcol(kernel_inv)]
    last_non_zero_rows = [findlast(x->abs(x)>0, col) for col in eachcol(kernel_inv)]
    @assert reduce(&, first_non_zero_rows .!= last_non_zero_rows)

    
    kernel_ratios = [kernel_inv[i[2], i[1]] for i in enumerate(last_non_zero_rows)] ./ [kernel_inv[i[2], i[1]] for i in enumerate(first_non_zero_rows)]

    reordered_sorted_col = fill(Int[], length(sorted_col))
    for j in 1:dim_o
        col_cluster = segment_wise_norm(vecs[:, sorted_col[j]], ranges)
        enum_cols = enumerate(eachcol(col_cluster))

        show(stdout, "text/plain", sorted_col[j])
        println()
        show(stdout, "text/plain", vecs[:, sorted_col[j]])
        println()
        show(stdout, "text/plain", col_cluster)
        println()
        
        cluster_ratios = [col_cluster[i[2], i[1]] for i in enumerate(first_non_zero_rows)] ./ [col_cluster[i[2], i[1]] for i in enumerate(last_non_zero_rows)]
        cluster_group = classify_by_profile(enum_cols, kernel_profile, ptol)
    
        ratio_matrix = kernel_ratios * cluster_ratios'
        show(stdout, "text/plain", cluster_ratios)
        println()
        show(stdout, "text/plain", kernel_ratios)
        println()
        show(stdout, "text/plain", ratio_matrix)
        println()
        reordered_col = match_making(cluster_group, kernel_group, ratio_matrix, rtol)
        reordered_sorted_col[j] = sorted_col[j][reordered_col]
    end
    
    return vecs[:, cat_rows(reduce(hcat, reordered_sorted_col))]
end

end # module