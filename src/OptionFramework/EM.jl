module EM
using Base.Iterators
using LinearAlgebra

"""
Suppose the input is a series of (state, action) tupples
  and the current policy matrices
  * pilo[o1, a1, s1] = P(a1 | o1, s1)
  * pihi[o1, o2, s2] = P(o2 | o1, s2)
return the series alpha where
  alpha[t][o] = P(s[2:t], a[1:t], O_t = o)
"""
function forward_recursion(series, pilo, pihi)
  (dim_o, dim_a, dim_s) = size(pilo)
  unit_vector = vcat([1], zeros(dim_o - 1))

  (s1, a1) = series[1]
  alpha = [normalize(unit_vector * pilo[1, a1, s1], 1)]

  for i in 2:length(series)
    (s, a) = series[i]
    alp = (pihi[:, :, s]' * alpha[i - 1]) .* pilo[:, a, s]
    push!(alpha, normalize(alp))
  end
  return alpha
end

"""
Suppose the input is a series of (state, action) tupples
  and the current policy matrices
  * pilo[o1, a1, s1] = P(a1 | o1, s1)
  * pihi[o1, o2, s2] = P(o2 | o1, s2)
return the series beta where
beta[t][o] = P(s[t+1:T], a[t+1:T] | O_t = o, s_t, a_t)
"""
function backward_recursion(series, pilo, pihi)
  (dim_o, dim_a, dim_s) = size(pilo)
  beta = [ones(dim_o) / 2 / dim_o]
  for i in 2:length(series)

  end
end

end
