module Utils

export dense_block_diag, extract_block_diag, cat_rows, cat_cols
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

end # module