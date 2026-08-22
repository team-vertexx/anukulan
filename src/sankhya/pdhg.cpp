#include "sankhya/pdhg.hpp"

#include <chrono>
#include <cmath>
#include <algorithm>
#include <cstdio>

namespace sankhya {
namespace {

double dot(const std::vector<double>& a, const std::vector<double>& b) {
  double sum = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) sum += a[i] * b[i];
  return sum;
}

double two_norm(const std::vector<double>& v) { return std::sqrt(dot(v, v)); }

// Bound multipliers implied by a dual point: lambda = c - K' y.
//
// The dual is feasible when each lambda_j can be absorbed by that variable's
// bounds. A variable bounded on both sides absorbs any sign; one bounded only
// from below needs lambda_j >= 0; only from above needs lambda_j <= 0; a free
// variable needs lambda_j = 0. Whatever is left over is the dual residual.
void reduced_costs(const StandardLp& lp, const std::vector<double>& y,
                   std::vector<double>* lambda, std::vector<double>* scratch) {
  scratch->resize(sz(lp.num_cols()));
  lp.kt.multiply(y.data(), scratch->data());
  lambda->resize(sz(lp.num_cols()));
  for (std::size_t j = 0; j < lambda->size(); ++j) (*lambda)[j] = lp.c[j] - (*scratch)[j];
}

}  // namespace

std::string to_string(PdhgStatus status) {
  switch (status) {
    case PdhgStatus::kOptimal:
      return "optimal";
    case PdhgStatus::kIterationLimit:
      return "iteration limit";
    case PdhgStatus::kTimeLimit:
      return "time limit";
    case PdhgStatus::kNumericalError:
      return "numerical error";
  }
  return "unknown";
}

PdhgResidual evaluate_residual(const StandardLp& lp, const std::vector<double>& x,
                               const std::vector<double>& y) {
  PdhgResidual r;
  std::vector<double> scratch;

  lp.primal_residual(x, &scratch, &r.primal_residual, nullptr);

  std::vector<double> lambda;
  std::vector<double> work;
  reduced_costs(lp, y, &lambda, &work);

  double dual_sq = 0.0;
  double bound_term = 0.0;
  for (std::size_t j = 0; j < lambda.size(); ++j) {
    const bool has_lo = lp.lower[j] > -kInf;
    const bool has_hi = lp.upper[j] < kInf;
    const double l = lambda[j];

    double leftover = 0.0;
    if (has_lo && has_hi) {
      leftover = 0.0;  // a boxed variable absorbs any sign
    } else if (has_lo) {
      leftover = std::fmin(l, 0.0);
    } else if (has_hi) {
      leftover = std::fmax(l, 0.0);
    } else {
      leftover = l;  // free variables need a zero reduced cost
    }
    dual_sq += leftover * leftover;

    // The bound contribution to the dual objective: minimising lambda_j * x_j
    // over the variable's own interval picks whichever end the sign favours.
    const double absorbed = l - leftover;
    if (absorbed > 0.0) {
      bound_term += absorbed * lp.lower[j];
    } else if (absorbed < 0.0) {
      bound_term += absorbed * lp.upper[j];
    }
  }
  r.dual_residual = std::sqrt(dual_sq);

  r.primal_objective = lp.standard_objective(x);
  r.dual_objective = dot(lp.q, y) + bound_term;
  r.absolute_gap = std::fabs(r.primal_objective - r.dual_objective);

  // Relative measures, as in PDLP: each residual against the natural scale of
  // the data it is compared with, so a tolerance means the same thing on a
  // problem whose numbers are large and one whose numbers are small.
  r.relative_primal = r.primal_residual / (1.0 + two_norm(lp.q));
  r.relative_dual = r.dual_residual / (1.0 + two_norm(lp.c));
  r.relative_gap = r.absolute_gap / (1.0 + std::fabs(r.primal_objective) +
                                     std::fabs(r.dual_objective));
  return r;
}

double estimate_matrix_norm(const StandardLp& lp, int iterations, double tolerance) {
  const Int n = lp.num_cols();
  const Int m = lp.num_rows();
  if (n == 0 || m == 0 || lp.k.nnz() == 0) return 0.0;

  // Power iteration on K'K. A deterministic non-uniform start avoids landing on
  // a vector that happens to be orthogonal to the leading singular vector.
  std::vector<double> v(sz(n));
  for (Int j = 0; j < n; ++j) v[sz(j)] = 1.0 + 0.1 * static_cast<double>(j % 7);
  double norm = two_norm(v);
  if (norm == 0.0) return 0.0;
  for (double& value : v) value /= norm;

  std::vector<double> kv(sz(m));
  std::vector<double> ktkv(sz(n));
  double sigma = 0.0;
  for (int iteration = 0; iteration < iterations; ++iteration) {
    lp.k.multiply(v.data(), kv.data());
    lp.kt.multiply(kv.data(), ktkv.data());
    const double next = two_norm(ktkv);
    if (next == 0.0) return 0.0;
    for (std::size_t j = 0; j < ktkv.size(); ++j) v[j] = ktkv[j] / next;
    const double estimate = std::sqrt(next);
    if (iteration > 0 && std::fabs(estimate - sigma) <= tolerance * estimate) {
      sigma = estimate;
      break;
    }
    sigma = estimate;
  }
  return sigma;
}

namespace {

// The weighted norm PDLP works in: ||(x, y)||_omega^2 = omega ||x||^2 + ||y||^2 / omega.
double weighted_norm_squared(const std::vector<double>& dx, const std::vector<double>& dy,
                             double omega) {
  return omega * dot(dx, dx) + dot(dy, dy) / omega;
}

// The restart criterion. PDLP restarts on a normalized duality gap, which needs a
// trust-region subproblem; cuPDLP replaces it with this weighted KKT error
// because it is a handful of reductions and so suits a GPU. Same three terms:
// primal infeasibility, dual infeasibility, and the duality gap.
struct KktWork {
  std::vector<double> kx;
  std::vector<double> kty;
};

double weighted_kkt_error(const StandardLp& lp, const std::vector<double>& x,
                          const std::vector<double>& y, double omega, KktWork* work) {
  work->kx.resize(sz(lp.num_rows()));
  work->kty.resize(sz(lp.num_cols()));
  lp.k.multiply(x.data(), work->kx.data());
  lp.kt.multiply(y.data(), work->kty.data());

  double primal_sq = 0.0;
  for (Int i = 0; i < lp.num_rows(); ++i) {
    const std::size_t si = sz(i);
    const double v = (i < lp.num_equalities)
                         ? (work->kx[si] - lp.q[si])
                         : std::fmin(work->kx[si] - lp.q[si], 0.0);
    primal_sq += v * v;
  }

  double dual_sq = 0.0;
  double bound_term = 0.0;
  for (Int j = 0; j < lp.num_cols(); ++j) {
    const std::size_t sj = sz(j);
    const double lambda = lp.c[sj] - work->kty[sj];
    const bool has_lo = lp.lower[sj] > -kInf;
    const bool has_hi = lp.upper[sj] < kInf;
    double leftover = 0.0;
    if (has_lo && has_hi) {
      leftover = 0.0;
    } else if (has_lo) {
      leftover = std::fmin(lambda, 0.0);
    } else if (has_hi) {
      leftover = std::fmax(lambda, 0.0);
    } else {
      leftover = lambda;
    }
    dual_sq += leftover * leftover;
    const double absorbed = lambda - leftover;
    if (absorbed > 0.0) {
      bound_term += absorbed * lp.lower[sj];
    } else if (absorbed < 0.0) {
      bound_term += absorbed * lp.upper[sj];
    }
  }

  const double gap = lp.standard_objective(x) - (dot(lp.q, y) + bound_term);
  const double value = omega * omega * primal_sq + dual_sq / (omega * omega) + gap * gap;
  return std::sqrt(value);
}

}  // namespace

PdhgResult solve_pdhg(const StandardLp& original, const PdhgOptions& options) {
  const auto start_time = std::chrono::steady_clock::now();
  auto elapsed = [&start_time]() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time)
        .count();
  };

  PdhgResult result;
  const Int n = original.num_cols();
  const Int m = original.num_rows();

  // Everything below runs on a scaled copy; convergence is judged on the
  // original, because that is the problem that was asked about.
  StandardLp lp = original;
  result.scaling = scale_lp(&lp, options.scaling);

  const double matrix_norm = estimate_matrix_norm(lp, 200, 1e-8);
  result.matrix_norm_estimate = matrix_norm;
  if (matrix_norm <= 0.0) {
    // No constraint matrix content: the answer is whatever the bounds and the
    // objective sign say, componentwise.
    result.x.assign(sz(n), 0.0);
    for (Int j = 0; j < n; ++j) {
      const double lo = original.lower[sz(j)];
      const double hi = original.upper[sz(j)];
      const double cj = original.c[sz(j)];
      double value = 0.0;
      if (cj > 0.0) {
        value = lo;
      } else if (cj < 0.0) {
        value = hi;
      } else {
        value = std::isfinite(lo) ? lo : (std::isfinite(hi) ? hi : 0.0);
      }
      result.x[sz(j)] = std::isfinite(value) ? value : 0.0;
    }
    result.y.assign(sz(m), 0.0);
    result.residual = evaluate_residual(original, result.x, result.y);
    result.objective = original.model_objective(result.x);
    result.status = result.residual.converged(options.tolerance)
                        ? PdhgStatus::kOptimal
                        : PdhgStatus::kNumericalError;
    result.message = "no constraint matrix content";
    result.solve_seconds = elapsed();
    return result;
  }

  // Step size stays below 1 / ||K||. The primal weight splits it between the two
  // steps: tau = eta / omega, sigma = eta * omega.
  double eta = 0.9 / matrix_norm;
  double omega = 1.0;
  if (options.primal_weight_updates) {
    const double c_norm = two_norm(lp.c);
    const double q_norm = two_norm(lp.q);
    if (c_norm > 1e-12 && q_norm > 1e-12) omega = c_norm / q_norm;
  }

  std::vector<double> x(sz(n), 0.0);
  std::vector<double> y(sz(m), 0.0);
  for (Int j = 0; j < n; ++j) {
    // Start inside the box, or the first projection does all the work.
    const double lo = lp.lower[sz(j)];
    const double hi = lp.upper[sz(j)];
    double v = 0.0;
    if (v < lo) v = std::isfinite(lo) ? lo : 0.0;
    if (v > hi) v = std::isfinite(hi) ? hi : 0.0;
    x[sz(j)] = v;
  }

  std::vector<double> x_next(sz(n), 0.0);
  std::vector<double> y_next(sz(m), 0.0);
  std::vector<double> kt_y(sz(n), 0.0);
  std::vector<double> x_bar(sz(n), 0.0);
  std::vector<double> k_x_bar(sz(m), 0.0);
  std::vector<double> k_x(sz(m), 0.0);
  std::vector<double> dx(sz(n), 0.0);
  std::vector<double> dy(sz(m), 0.0);
  std::vector<double> k_dx(sz(m), 0.0);

  lp.k.multiply(x.data(), k_x.data());

  // Running average over the current restart period, weighted by step size, and
  // the anchors the restart rules compare against.
  std::vector<double> sum_x(sz(n), 0.0);
  std::vector<double> sum_y(sz(m), 0.0);
  double sum_weight = 0.0;
  std::vector<double> avg_x(sz(n), 0.0);
  std::vector<double> avg_y(sz(m), 0.0);

  std::vector<double> restart_x = x;
  std::vector<double> restart_y = y;
  std::vector<double> previous_restart_x = x;
  std::vector<double> previous_restart_y = y;
  bool have_previous_restart = false;

  KktWork kkt_work;
  double kkt_at_restart = weighted_kkt_error(lp, x, y, omega, &kkt_work);
  double last_candidate_kkt = kInf;

  std::vector<double> unscaled_x;
  std::vector<double> unscaled_y;

  const double tolerance = options.tolerance;
  Int iteration = 0;
  Int iterations_since_restart = 0;
  PdhgStatus status = PdhgStatus::kIterationLimit;

  for (; iteration < options.max_iterations; ++iteration) {
    lp.kt.multiply(y.data(), kt_y.data());

    // One PDHG step, retried at a smaller step size if the adaptive rule rejects
    // it. Rejected trials cost one matrix-vector product each.
    for (int attempt = 0;; ++attempt) {
      const double tau = eta / omega;
      const double sigma = eta * omega;

      for (Int j = 0; j < n; ++j) {
        const std::size_t sj = sz(j);
        double v = x[sj] - tau * (lp.c[sj] - kt_y[sj]);
        if (v < lp.lower[sj]) v = lp.lower[sj];
        if (v > lp.upper[sj]) v = lp.upper[sj];
        x_next[sj] = v;
        dx[sj] = v - x[sj];
        x_bar[sj] = v + dx[sj];  // 2 x_next - x, the extrapolated point
      }

      lp.k.multiply(x_bar.data(), k_x_bar.data());

      for (Int i = 0; i < m; ++i) {
        const std::size_t si = sz(i);
        double v = y[si] + sigma * (lp.q[si] - k_x_bar[si]);
        if (i >= lp.num_equalities && v < 0.0) v = 0.0;
        y_next[si] = v;
        dy[si] = v - y[si];
        // K x_bar = 2 K x_next - K x, so K (x_next - x) comes out of the two
        // products already computed - no extra matrix-vector product needed.
        k_dx[si] = 0.5 * (k_x_bar[si] - k_x[si]);
      }

      if (!options.adaptive_step_size) break;

      // PDLP's rule: the step is safe while eta is below the local curvature
      // limit eta_bar, and the new step size may fall fast but grow only slowly.
      // The magnitude matters, not the sign: the step is safe while
      // eta * |dy' K dx| stays under half the weighted movement. Dropping the
      // absolute value here lets eta grow without bound whenever the
      // interaction happens to be negative, and the iterates blow up.
      const double interaction = std::fabs(dot(dy, k_dx));
      const double movement = weighted_norm_squared(dx, dy, omega);
      double eta_bar = kInf;
      if (interaction > 0.0) eta_bar = movement / (2.0 * interaction);

      const double k1 = static_cast<double>(iteration + 1);
      double eta_next = eta;
      if (std::isfinite(eta_bar)) {
        eta_next = std::fmin((1.0 - std::pow(k1, -0.3)) * eta_bar,
                             (1.0 + std::pow(k1, -0.6)) * eta);
      } else {
        eta_next = (1.0 + std::pow(k1, -0.6)) * eta;
      }
      if (!(eta_next > 0.0) || !std::isfinite(eta_next)) eta_next = eta;

      const bool accept = (eta <= eta_bar);
      eta = eta_next;
      if (accept) break;
      if (attempt > 60) break;  // give up rejecting and take the step
    }

    for (Int i = 0; i < m; ++i) k_x[sz(i)] = 0.5 * (k_x_bar[sz(i)] + k_x[sz(i)]);
    x.swap(x_next);
    y.swap(y_next);
    ++iterations_since_restart;

    if (options.restarts) {
      for (Int j = 0; j < n; ++j) sum_x[sz(j)] += eta * x[sz(j)];
      for (Int i = 0; i < m; ++i) sum_y[sz(i)] += eta * y[sz(i)];
      sum_weight += eta;
    }

    const bool check = ((iteration + 1) % options.termination_check_frequency == 0);
    if (!check) continue;

    // The candidate is whichever of the current iterate and the running average
    // has the smaller KKT error. Both restarting and stopping are judged on it.
    const double current_kkt = weighted_kkt_error(lp, x, y, omega, &kkt_work);
    bool use_average = false;
    double candidate_kkt = current_kkt;
    if (options.restarts && sum_weight > 0.0) {
      for (Int j = 0; j < n; ++j) avg_x[sz(j)] = sum_x[sz(j)] / sum_weight;
      for (Int i = 0; i < m; ++i) avg_y[sz(i)] = sum_y[sz(i)] / sum_weight;
      const double average_kkt = weighted_kkt_error(lp, avg_x, avg_y, omega, &kkt_work);
      if (average_kkt < current_kkt) {
        use_average = true;
        candidate_kkt = average_kkt;
      }
    }
    const std::vector<double>& cand_x = use_average ? avg_x : x;
    const std::vector<double>& cand_y = use_average ? avg_y : y;

    result.scaling.scaling.unscale_primal(cand_x, &unscaled_x);
    result.scaling.scaling.unscale_dual(cand_y, &unscaled_y);
    const PdhgResidual r = evaluate_residual(original, unscaled_x, unscaled_y);

    if (!std::isfinite(r.primal_residual) || !std::isfinite(r.dual_residual)) {
      status = PdhgStatus::kNumericalError;
      result.message = "iterates stopped being finite";
      ++iteration;
      x = cand_x;
      y = cand_y;
      break;
    }
    if (r.converged(tolerance)) {
      status = PdhgStatus::kOptimal;
      ++iteration;
      x = cand_x;
      y = cand_y;
      break;
    }
    if (options.verbose && (iteration + 1) % options.log_frequency == 0) {
      std::printf(
          "  iter %7d  primal %.3e  dual %.3e  gap %.3e  kkt %.3e  eta %.3e  w %.3e\n",
          iteration + 1, r.relative_primal, r.relative_dual, r.relative_gap,
          candidate_kkt, eta, omega);
    }
    if (elapsed() > options.time_limit_seconds) {
      status = PdhgStatus::kTimeLimit;
      ++iteration;
      x = cand_x;
      y = cand_y;
      break;
    }

    if (!options.restarts) continue;

    // cuPDLP's three restart conditions, with its constants.
    const bool sufficient = candidate_kkt <= 0.2 * kkt_at_restart;
    const bool necessary = candidate_kkt <= 0.8 * kkt_at_restart &&
                           candidate_kkt > last_candidate_kkt;
    const bool too_long =
        static_cast<double>(iterations_since_restart) >= 0.36 * static_cast<double>(iteration + 1);
    last_candidate_kkt = candidate_kkt;

    if (!(sufficient || necessary || too_long)) continue;

    x = cand_x;
    y = cand_y;
    lp.k.multiply(x.data(), k_x.data());

    if (options.primal_weight_updates && have_previous_restart) {
      // Exponential smoothing in log space, theta = 0.5, so the new weight is
      // the geometric mean of the old one and the ratio of how far the primal
      // and dual moved over the last restart period.
      double delta_x_sq = 0.0;
      double delta_y_sq = 0.0;
      for (Int j = 0; j < n; ++j) {
        const double d = x[sz(j)] - previous_restart_x[sz(j)];
        delta_x_sq += d * d;
      }
      for (Int i = 0; i < m; ++i) {
        const double d = y[sz(i)] - previous_restart_y[sz(i)];
        delta_y_sq += d * d;
      }
      const double delta_x = std::sqrt(delta_x_sq);
      const double delta_y = std::sqrt(delta_y_sq);
      if (delta_x > 1e-12 && delta_y > 1e-12) {
        const double candidate = std::exp(0.5 * std::log(delta_y / delta_x) +
                                          0.5 * std::log(omega));
        if (std::isfinite(candidate) && candidate > 1e-12 && candidate < 1e12) {
          omega = candidate;
        }
      }
    }

    previous_restart_x = restart_x;
    previous_restart_y = restart_y;
    have_previous_restart = true;
    restart_x = x;
    restart_y = y;

    std::fill(sum_x.begin(), sum_x.end(), 0.0);
    std::fill(sum_y.begin(), sum_y.end(), 0.0);
    sum_weight = 0.0;
    iterations_since_restart = 0;
    last_candidate_kkt = kInf;
    kkt_at_restart = weighted_kkt_error(lp, x, y, omega, &kkt_work);
    ++result.restarts;
  }

  result.scaling.scaling.unscale_primal(x, &result.x);
  result.scaling.scaling.unscale_dual(y, &result.y);
  result.residual = evaluate_residual(original, result.x, result.y);
  result.objective = original.model_objective(result.x);
  result.iterations = iteration;
  result.status = status;
  result.final_step_size = eta;
  result.final_primal_weight = omega;
  result.solve_seconds = elapsed();
  return result;
}

}  // namespace sankhya
