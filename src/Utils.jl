module Utils

export dense_block_diag, extract_block_diag, cat_rows, cat_cols, submat
using LinearAlgebra
using SparseArrays

function dense_block_diag(ms)
    Matrix(blockdiag(sparse.(ms)...))
end

function extract_block_diag(m, nrow, ncol, ndep)
    mask = .!iszero.(kron(I(ndep), ones(nrow, ncol)))
    reshape(m[mask], nrow, ncol, ndep)
end

function cat_rows(m)
    vcat(eachrow(m)...)
end

function cat_cols(m)
    vcat(eachcol(m)...)
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

end # module