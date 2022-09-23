module Utils

export splat, vectorize, dense_block_diag, extract_block_diag, cat_rows, cat_cols, submat, 
    uniform_partition_indices, uniform_partition_views, uniform_partitions, 
    pretty_println, hungarian_algorithm, softmax
using LinearAlgebra
using SparseArrays
using Base.Iterators

splat(f::Function) = args->f(args...)
vectorize(f::Function) = (args...)->f.(args...)

function dense_block_diag(ms)
    (nrow, ncol) = size(first(ms))
    ndep = length(ms)
    block_diag = zeros(eltype(first(ms)), nrow * ndep, ncol * ndep)
    partition_views = uniform_partition_views(block_diag, nrow, ncol)
    for (i, m) in enumerate(ms)
        partition_views[i, i] .= m
    end
    return block_diag
end

function extract_block_diag(m, nrow, ncol)
    partition_indices = uniform_partition_indices(m, nrow, ncol)
    (ndep, ndep_test) = size(partition_indices)
    @assert ndep == ndep_test

    diag_blocks = [m[partition_indices[i, i]...] for i in 1:ndep]
    dense_tensor = Array{Float64, 3}(undef, nrow, ncol, ndep)
    for i in 1:ndep
        dense_tensor[:, :, i] = diag_blocks[i] 
    end
    return dense_tensor
end

function cat_rows(m)
    return reduce(vcat, eachrow(m))
end

function cat_cols(m)
    return reduce(vcat, eachcol(m))
end

function submat(m, s2, s3, stride_r, stride_c)
    head2 = (s2 - 1) * stride_r + 1
    tail2 = head2 + stride_r - 1
    
    head3 = (s3 - 1) * stride_c + 1
    tail3 = head3 + stride_c - 1
    
    m[head2:tail2, head3:tail3]
end

function subind(s2, s3, stride_r, stride_c)
    head2 = (s2 - 1) * stride_r + 1
    tail2 = head2 + stride_r - 1
    
    head3 = (s3 - 1) * stride_c + 1
    tail3 = head3 + stride_c - 1
    
    (head2:tail2, head3:tail3)
end

function uniform_partition_indices(tensor, sub_dim...)
    tensor_dim = size(tensor)
    num_dims = length(tensor_dim)

    # sanity check
    @assert num_dims == length(sub_dim)
    for i in 1:num_dims
        @assert rem(tensor_dim[i], sub_dim[i]) == 0
    end

    # dimension of the resulting tensor in terms of blocks
    partition_dim = splat(div).(zip(tensor_dim, sub_dim))
    
    # generating ranges
    starting_locations(stride, num_strides) = range(start=1, step=stride, length=num_strides)
    ending_locations(stride, num_strides) = range(start=stride, step=stride, length=num_strides)

    part_starts = splat(starting_locations).(zip(sub_dim, partition_dim))
    part_ends = splat(ending_locations).(zip(sub_dim, partition_dim))

    start_end_pairs = splat(zip).(zip(part_starts, part_ends))

    convert_to_ranges(se_pairs) = splat(range).(se_pairs)
    partition_index_grid = convert_to_ranges.(product(start_end_pairs...))

    return partition_index_grid
end

function uniform_partition_views(tensor, sub_dim...)
    partition_index_grid = uniform_partition_indices(tensor, sub_dim...)

    get_sub_tensor_view(ranges) = view(tensor, ranges...)
    return get_sub_tensor_view.(partition_index_grid)
end

function uniform_partitions(tensor, sub_dim...) 
    partition_index_grid = uniform_partition_indices(tensor, sub_dim...)

    get_sub_tensor(ranges) = tensor[ranges...]
    return get_sub_tensor.(partition_index_grid)
end

function pretty_println(io::IO, stuff; kwargs...)
    limit = get(kwargs, :limit, true)
    compact = get(kwargs, :compact, true)

    show(IOContext(io, :limit=>limit, :compact=>compact), "text/plain", stuff)
    println(io)
end
pretty_println(stuff; kwargs...) = pretty_println(stdout, stuff; kwargs...)

function cardinal_complement(indices, n)
    solution = Vector{Int}(undef, n - length(indices))
    bitstring = trues(n)
    bitstring[indices] .= false
    i = 1
    for j in 1:n
        if bitstring[j]
            solution[i] = j
            i += 1
        end
    end
    return solution
end

function mark_matrix(m)
    (rows, cols) = size(m)
    @assert rows == cols
    n = rows

    zero_mask = iszero.(m)

    uncovered_rows = trues(n)
    mandatory_rows = Int[]
    flex_col_row_pairs = zeros(Int, n)

    zeros_per_row = dropdims(sum(zero_mask, dims=2), dims=2)
    
    while true
        # compute the number of zero elements per row and remove rows that don't have any zero elements
        nzpr = collect(Iterators.filter(x -> last(x) > 0, enumerate(zeros_per_row))) 
        length(nzpr) == 0 && break

        (rwmz, _) = argmin(x -> last(x), nzpr) # (row with minimum number of zeros, row_index)

        zor = findall(zero_mask[rwmz, :]) # zero elements on row

        column_value = dropdims(sum(zero_mask[:, zor], dims=1), dims=1)
        markable_cols = any(column_value .>= length(zor))
        if !markable_cols
            push!(mandatory_rows, rwmz)
        else
            lowest_val_col = zor[argmin(column_value)]
            flex_col_row_pairs[lowest_val_col] = rwmz

            zeros_per_row -= zero_mask[:, lowest_val_col]
            zero_mask[:, lowest_val_col] .= false # block off column
        end
        zeros_per_row[rwmz] = 0
        zero_mask[rwmz, :] .= false # block off row
        uncovered_rows[rwmz] = false # row is now covered
    end

    zero_mask = iszero.(m) # refresh zero_mask
    covered_columns = flex_col_row_pairs .> 0

    mandatory_columns = dropdims(sum(view(zero_mask, uncovered_rows, :), dims=1), dims=1) .> 0
    lost_rows = flex_col_row_pairs[mandatory_columns] 
    while !isempty(lost_rows)
        incidental_mandatory_columns = dropdims(sum(view(zero_mask, lost_rows, :), dims=1), dims=1) .> 0
        new_mandatory_columns = .!mandatory_columns .& incidental_mandatory_columns

        mandatory_columns .|= new_mandatory_columns
        lost_rows = flex_col_row_pairs[new_mandatory_columns] 
    end

    marked_rows = flex_col_row_pairs[.!mandatory_columns .& covered_columns]
    append!(marked_rows, mandatory_rows)

    marked_cols = findall(mandatory_columns)
    return marked_rows, marked_cols
end

function adjust_matrix!(m, marked_rows, marked_cols)
    (rows, cols) = size(m)
    unmarked_rows = cardinal_complement(marked_rows, rows)
    unmarked_cols = cardinal_complement(marked_cols, cols)

    unmarked_region = view(m, unmarked_rows, unmarked_cols)
    doubly_marked_region = view(m, marked_rows, marked_cols)

    unmarked_minimum = minimum(unmarked_region)
    unmarked_region .-= unmarked_minimum
    doubly_marked_region .+= unmarked_minimum
end

function hungarian_algorithm(mat)
    (rows, cols) = size(mat)
    @assert rows == cols
    n = rows

    m = deepcopy(mat)
    m .= m .- minimum(m, dims=2)
    m .= m .- minimum(m, dims=1)
    for i in 1:(n+n)
        (marked_rows, marked_cols) = mark_matrix(m)
        length(marked_rows) == n && return marked_rows

        adjust_matrix!(m, marked_rows, marked_cols)
    end
    error("should not have reached here")
end

function softmax(vec)
    e_vec = exp.(vec)
    return e_vec ./ sum(e_vec)
end

end
