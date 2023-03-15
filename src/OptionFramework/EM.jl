module EM
export iterate_policy, iterate_and_print

using Base.Iterators
using LinearAlgebra
using ..OptionFramework
using ...Utils

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
    push!(alpha, normalize(alp, 1))
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
  beta = [ones(dim_o) / dim_o]
  T = length(series)
  for i in 2:T
    t = T - i + 1
    (s_next, a_next) = series[t + 1]
    bet = pihi[:, :, s_next] * (beta[i - 1] .* pilo[:, a_next, s_next])
    push!(beta, normalize(bet, 1))
  end
  return reverse(beta)
end

"""
Then input is the alpha and beta series output from the functions above
  * alpha[t][o] = P(s[2:t], a[1:t], O_t = o)
  * beta[t][o] = P(s[t+1:T], a[t+1:T] | O_t = o, s_t, a_t)
return the series gamma where
gamma[t][o] ~ alpha[t][o] * beta[t][o]
"""
function smoothing(alpha, beta)
  vectorize(f) = (x...) -> f.(x...)
  return normalize.(vectorize(*).(alpha, beta), 1)
end

"""
Suppose the input is a series of (state, action) tupples,
  the current policy matrices, and the forward-backward series alpha and beta
  * pilo[o1, a1, s1] = P(a1 | o1, s1)
  * pihi[o1, o2, s2] = P(o2 | o1, s2)
  * alpha[t][o] = P(s[2:t], a[1:t], O_t = o)
  * beta[t][o] = P(s[t+1:T], a[t+1:T] | O_t = o, s_t, a_t)
return the series gamma_tilde
"""
function smoothing_tilde(series, pilo, pihi, alpha, beta)
  (dim_o, dim_a, dim_s) = size(pilo)
  gamma_tilde = []
  T = length(series)
  for i in 1:T-1
    (s, a) = series[i + 1]
    alp = alpha[i]
    bet = beta[i + 1]
    gam_ti = pihi[:, :, s] .* (alp * (pilo[:, a, s] .* bet)')
    push!(gamma_tilde, normalize(gam_ti, 1))
  end
  return gamma_tilde
end

"""
Performs a single Expectation-Maximization step
"""
function iterate_policy(series, pilo, pihi)
  (dim_o, dim_a, dim_s) = size(pilo)
  alpha = forward_recursion(series, pilo, pihi)
  beta = backward_recursion(series, pilo, pihi)
  gamma = smoothing(alpha, beta)
  gamma_tilde = smoothing_tilde(series, pilo, pihi, alpha, beta)

  new_pihi = zeros(dim_o, dim_o, dim_s)
  new_pilo = zeros(dim_o, dim_a, dim_s)
  T = length(series)

  for s in 1:dim_s
    relevant_t = filter(t -> series[t][1] == s, 2:T) .- 1
    new_pihi[:, :, s] = sum(gamma_tilde[relevant_t]) ./ sum(gamma[relevant_t] .* [ones(dim_o)'])

    relevant_t_s = filter(t -> series[t][1] == s, 1:T)
    sum_gamma_s = sum(gamma[relevant_t_s])
    for a in 1:dim_a
      relevant_t_sa = filter(t -> series[t] == (s, a), 1:T)
      new_pilo[:, a, s] = sum(gamma[relevant_t_sa]) ./ sum_gamma_s
    end
  end
  return (new_pilo, new_pihi)
end

"""
Iterate until the error reach a certain threshold
"""
function iterate_and_print(problem, raw_3rd_order, n_sample, pilo, pihi; T=500)
  (true_pihi_kron, true_pilo_kron, true_phi_kron) = dense_block_diag.(eachslice.(problem(), dims=3))
  pilo_kron = dense_block_diag(eachslice(pilo, dims=3))
  pihi_kron = dense_block_diag(eachslice(pihi, dims=3))
  series = raw_3rd_order.sample_path[1:n_sample]
  dim_o = size(pilo)[1]

  error() = begin
    err_lo = compare_pilo(pilo_kron, true_pilo_kron, dim_o)
    err_hi = compare_pihi(pihi_kron, true_pihi_kron, dim_o)
    return sqrt(err_lo^2 + err_hi^2)
  end

  for t in 1:T
    (pilo, pihi) = iterate_policy(series, pilo, pihi)
    pilo_kron = dense_block_diag(eachslice(pilo, dims=3))
    pihi_kron = dense_block_diag(eachslice(pihi, dims=3))
    println(t, " ", error())
  end
  # pretty_println(pilo_kron)
  # pretty_println(true_pilo_kron)
  # pretty_println(pihi_kron)
  # pretty_println(true_pihi_kron)
  return (pilo, pihi)
end
end # module
