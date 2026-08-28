#include "sankhya/pdhg.hpp"

#include "sankhya/backend.hpp"

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

double inf_norm(const std::vector<double>& v) {
  double m = 0.0;
  for (const double x : v) m = std::fmax(m, std::fabs(x));
  return m;
}

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
    case PdhgStatus::kPrimalInfeasible:
      return "primal infeasible";
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

  lp.primal_residual(x, &scratch, &r.primal_residual, &r.primal_residual_inf);

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
    r.dual_residual_inf = std::fmax(r.dual_residual_inf, std::fabs(leftover));

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
  r.relative_primal_inf = r.primal_residual_inf / (1.0 + inf_norm(lp.q));
  r.relative_dual_inf = r.dual_residual_inf / (1.0 + inf_norm(lp.c));
  r.relative_gap = r.absolute_gap / (1.0 + std::fabs(r.primal_objective) +
                                     std::fabs(r.dual_objective));
  return r;
}

// Farkas test for an empty feasible set.
//
// The set {x : K_eq x = q_eq, K_ineq x >= q_ineq, l <= x <= u} is empty when
// there is a y, non-negative on the inequality block, with
//
//     max_{l <= x <= u} y'Kx  <  y'q
//
// The maximum on the left is separable: with lambda = K'y it is the sum over
// columns of lambda_j u_j where lambda_j > 0 and lambda_j l_j where it is
// negative. So the whole test is one transpose product and a pass over the
// columns.
//
// PDLP's observation is that while the iterates themselves diverge on an
// infeasible problem, the difference of consecutive dual iterates converges to
// exactly such a ray, so it is the natural thing to test.
bool is_infeasibility_certificate(const StandardLp& lp, const std::vector<double>& dy,
                                  double tolerance, std::vector<double>* work) {
  double scale = 0.0;
  for (Int i = 0; i < lp.num_rows(); ++i) {
    if (i >= lp.num_equalities && dy[sz(i)] < 0.0) return false;  // outside the cone
    scale = std::fmax(scale, std::fabs(dy[sz(i)]));
  }
  if (scale <= 0.0) return false;

  work->resize(sz(lp.num_cols()));
  lp.kt.multiply(dy.data(), work->data());

  double support = 0.0;
  for (Int j = 0; j < lp.num_cols(); ++j) {
    const double lambda = (*work)[sz(j)];
    if (lambda > 0.0) {
      if (lp.upper[sz(j)] >= kInf) return false;  // unbounded above, no certificate
      support += lambda * lp.upper[sz(j)];
    } else if (lambda < 0.0) {
      if (lp.lower[sz(j)] <= -kInf) return false;
      support += lambda * lp.lower[sz(j)];
    }
  }

  double rhs = 0.0;
  for (Int i = 0; i < lp.num_rows(); ++i) rhs += lp.q[sz(i)] * dy[sz(i)];

  // Normalise by the size of the ray so the margin means the same thing whatever
  // scale the iterates happen to be at.
  return (rhs - support) / scale > tolerance;
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

namespace {

// The two sub-problems feasibility polishing solves. Both keep K untouched,
// which is what lets the sub-solve reuse the enclosing solve's matrix norm.
//
// Primal: drop the objective. Same feasible set, nothing pulling against it.
void make_primal_feasibility(StandardLp* lp) {
  std::fill(lp->c.begin(), lp->c.end(), 0.0);
  lp->objective_offset = 0.0;
}

// Dual: drop the right-hand side and the bound values, keeping which bounds are
// finite. The dual feasible set is {y >= 0 on the inequality block, with
// c - K'y respecting the sign each column's finite bounds impose}, and none of
// K, c or the finiteness pattern moves here - only the dual objective, which
// becomes identically zero. x = 0 is feasible for what is left, so the modified
// problem has optimal value zero and PDHG on it converges to a dual feasible y.
void make_dual_feasibility(StandardLp* lp) {
  std::fill(lp->q.begin(), lp->q.end(), 0.0);
  for (std::size_t j = 0; j < lp->lower.size(); ++j) {
    if (lp->lower[j] > -kInf) lp->lower[j] = 0.0;
    if (lp->upper[j] < kInf) lp->upper[j] = 0.0;
  }
  lp->objective_offset = 0.0;
}

// Depth of nested solves on this thread. The backend releases every uploaded
// matrix at once, so the sub-solves must not do it while the enclosing solve is
// still running.
thread_local int g_solve_depth = 0;

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
  const LinAlgBackend& backend =
      options.backend ? *options.backend : default_backend();
  StandardLp lp = original;
  result.scaling = scale_lp(&lp, options.scaling);

  backend.prepare(lp.k);
  backend.prepare(lp.kt);
  struct ReleaseGuard {
    const LinAlgBackend& backend;
    bool outermost;
    ~ReleaseGuard() {
      --g_solve_depth;
      if (outermost) backend.release();
    }
  } release_guard{backend, g_solve_depth == 0};
  ++g_solve_depth;

  // The sub-solves share the enclosing solve's matrix exactly, so they are
  // handed its norm rather than paying two hundred power iterations for the
  // same number.
  const double matrix_norm = options.known_matrix_norm > 0.0
                                 ? options.known_matrix_norm
                                 : estimate_matrix_norm(lp, 200, 1e-8);
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
    result.status = result.residual.converged(options.tolerance,
                                             options.absolute_tolerance,
                                             options.require_inf_norm_termination)
                        ? PdhgStatus::kOptimal
                        : PdhgStatus::kNumericalError;
    result.message = "no constraint matrix content";
    result.solve_seconds = elapsed();
    return result;
  }

  // Step size stays below 1 / ||K||. The primal weight splits it between the two
  // steps: tau = eta / omega, sigma = eta * omega.
  double eta = 0.9 / matrix_norm;
  // cuPDLPx fixes the step size rather than adapting it, at a scale just under
  // the 1/||K|| the convergence proof needs.
  const bool adaptive = options.adaptive_step_size && !options.constant_step_size;
  if (options.constant_step_size) eta = options.constant_step_scale / matrix_norm;
  double omega = 1.0;
  if (options.primal_weight_updates) {
    const double c_norm = two_norm(lp.c);
    const double q_norm = two_norm(lp.q);
    if (c_norm > 1e-12 && q_norm > 1e-12) omega = c_norm / q_norm;
  }
  // A polishing sub-solve is handed the values the main loop had reached. The
  // rule above cannot produce them: the sub-problem's c or q is all zeros, so
  // it would fall through to a weight of 1.
  if (options.initial_step_size > 0.0) eta = options.initial_step_size;
  if (options.initial_primal_weight > 0.0) omega = options.initial_primal_weight;

  std::vector<double> x(sz(n), 0.0);
  std::vector<double> y(sz(m), 0.0);
  const bool warm = options.warm_x != nullptr &&
                    options.warm_x->size() == sz(n) &&
                    options.warm_y != nullptr &&
                    options.warm_y->size() == sz(m);
  if (warm) {
    x = *options.warm_x;
    y = *options.warm_y;
    // The dual iterate lives in the cone the projection maintains, so a warm
    // start has to be put there before the loop assumes it already is.
    for (Int i = lp.num_equalities; i < m; ++i) {
      if (!(y[sz(i)] > 0.0)) y[sz(i)] = 0.0;
    }
  }
  for (Int j = 0; j < n; ++j) {
    // Start inside the box, or the first projection does all the work.
    const double lo = lp.lower[sz(j)];
    const double hi = lp.upper[sz(j)];
    double v = warm ? x[sz(j)] : 0.0;
    if (!std::isfinite(v)) v = 0.0;
    if (v < lo) v = std::isfinite(lo) ? lo : 0.0;
    if (v > hi) v = std::isfinite(hi) ? hi : 0.0;
    x[sz(j)] = v;
  }

  // The working set lives wherever the backend puts it. On CUDA that is the
  // device, and it stays there - the loop below never moves a vector across the
  // bus. Data comes back only at the convergence check, every fortieth
  // iteration, because that is the only place the host needs to look at it.
  //
  // The first CUDA backend took host pointers and staged every argument across
  // on every call, roughly ten transfers per iteration. Measured on a T4 it came
  // out slower than the CPU it was meant to accelerate. The kernels were never
  // the problem.
  std::vector<double> host_x = x;
  std::vector<double> host_y = y;
  std::vector<double> host_dy(sz(m), 0.0);
  std::vector<double> host_avg_x(sz(n), 0.0);
  std::vector<double> host_avg_y(sz(m), 0.0);

  BackendVector d_x(backend, n), x_next(backend, n), kt_y(backend, n);
  BackendVector x_bar(backend, n), dx(backend, n), sum_x(backend, n);
  BackendVector avg_x(backend, n);
  BackendVector d_y(backend, m), y_next(backend, m), k_x_bar(backend, m);
  BackendVector k_x(backend, m), dy(backend, m), k_dx(backend, m);
  BackendVector sum_y(backend, m), avg_y(backend, m);

  // Problem data the kernels read, uploaded once and left alone.
  BackendVector d_c(backend, n), d_lower(backend, n), d_upper(backend, n);
  BackendVector d_q(backend, m);
  d_c.upload(lp.c);
  d_lower.upload(lp.lower);
  d_upper.upload(lp.upper);
  d_q.upload(lp.q);
  d_x.upload(host_x);
  // The dual iterate needs uploading for the same reason the primal does. It
  // was missing here for as long as both started at zero, which hid two things:
  // a warm-started dual silently did nothing, and on CUDA the loop began from
  // whatever cudaMalloc handed back. The CPU allocator value-initialises, so
  // neither was visible on this machine.
  d_y.upload(host_y);
  // Same reason: these are accumulated into, not written, so they have to start
  // at zero rather than at whatever the allocator returned.
  backend.fill(sum_x.data(), n, 0.0);
  backend.fill(sum_y.data(), m, 0.0);

  backend.multiply(lp.k, d_x.data(), k_x.data());

  double sum_weight = 0.0;
  // The Halpern anchor: the point the current epoch started from.
  BackendVector anchor_x(backend, n), anchor_y(backend, m);
  // K times the anchor, kept alongside it. K is linear, so the blend below can
  // be applied to K z as well as to z, and the product never has to be redone.
  BackendVector anchor_kx(backend, m);
  bool anchor_needs_reset = true;
  Int halpern_k = 0;
  // ||z - T(z)||^2 in the weighted norm, and its value at the start of the
  // current epoch. The Halpern restart test is a decay of that quantity, which
  // the adaptive step size already computes.
  double movement = 0.0;
  double epoch_start_movement = 0.0;

  // Restart anchors are read only at restarts, which are rare, so they stay on
  // the host rather than occupying device memory.
  std::vector<double> restart_x = host_x;
  std::vector<double> restart_y = host_y;
  std::vector<double> previous_restart_x = host_x;
  std::vector<double> previous_restart_y = host_y;
  bool have_previous_restart = false;

  KktWork kkt_work;
  double kkt_at_restart = weighted_kkt_error(lp, x, y, omega, &kkt_work);
  double last_candidate_kkt = kInf;

  // The fixed-point residual at the start of the current epoch, and its value
  // at the previous check. Tracked separately from the Halpern anchor's copy,
  // which only exists when Halpern is on.
  double residual_at_restart = kInf;
  double last_residual = kInf;
  // PID state for the primal weight, carried across restarts.
  double weight_error_sum = 0.0;
  double weight_error_previous = 0.0;

  std::vector<double> unscaled_x;
  std::vector<double> unscaled_y;
  std::vector<double> infeasibility_work;

  const double tolerance = options.tolerance;
  const double gap_tolerance =
      options.gap_tolerance > 0.0 ? options.gap_tolerance : options.tolerance;

  // Halpern is a restart scheme, so it means nothing with restarts switched
  // off. The anchor is only ever reset inside the restart block, so
  // halpern-without-restarts sets one anchor on the first iteration and then
  // blends against it forever - neither Halpern nor plain PDHG, but a permanent
  // pull toward the starting point. The ablation that turns restarts off has to
  // turn this off with it, or it is not measuring what its name says.
  const bool halpern = options.halpern && options.restarts;

  // Reflection is half of a scheme, not a step on its own. R(z) overshoots the
  // operator, and what pulls it back is the Halpern anchor; without that this
  // is plain reflected PDHG, which does not converge - it cycles. The ablation
  // that turns restarts off has to turn this off with it, or it is measuring a
  // divergent iteration and calling it a baseline.
  const double reflection = halpern ? options.reflection : 0.0;

  // Which measures have to hold to stop. The polishing sub-solves ask for one
  // side only, since one side is all they are solving for.
  auto passes = [&](const PdhgResidual& r) {
    switch (options.termination) {
      case PdhgOptions::Termination::kPrimalFeasibility:
        return r.primal_ok(tolerance, options.absolute_tolerance,
                           options.require_inf_norm_termination);
      case PdhgOptions::Termination::kDualFeasibility:
        return r.dual_ok(tolerance, options.absolute_tolerance,
                         options.require_inf_norm_termination);
      case PdhgOptions::Termination::kFull:
        break;
    }
    return r.converged(tolerance, options.absolute_tolerance,
                       options.require_inf_norm_termination, gap_tolerance);
  };

  // Feasibility polishing state. The two sub-problems are built once and
  // reused: they share this solve's matrix, and the backend caches its uploads
  // by matrix address, so rebuilding them per attempt would re-upload K to the
  // device for nothing.
  StandardLp primal_feas_lp;
  StandardLp dual_feas_lp;
  bool feasibility_problems_built = false;
  std::vector<double> polished_x;
  std::vector<double> polished_y;
  PdhgResidual polished_residual;
  bool have_polished = false;
  Int next_polish_iteration = options.polish_first_iteration;

  // One polishing attempt. Takes a scaled primal-dual pair, returns the
  // unscaled polished pair and what it cost. Says nothing about whether the
  // result is worth having - that is the caller's test, and it is the same test
  // the caller would have had to pass anyway.
  auto run_polish = [&](const std::vector<double>& seed_x,
                        const std::vector<double>& seed_y, Int budget,
                        std::vector<double>* out_x, std::vector<double>* out_y,
                        PdhgResidual* out_r, Int* spent) {
    if (!feasibility_problems_built) {
      primal_feas_lp = lp;
      make_primal_feasibility(&primal_feas_lp);
      dual_feas_lp = lp;
      make_dual_feasibility(&dual_feas_lp);
      feasibility_problems_built = true;
    }

    PdhgOptions sub = options;
    sub.polish_feasibility = false;  // no recursion
    sub.polish_on_exit = false;
    sub.detect_infeasibility = false;  // both sub-problems are feasible by construction
    sub.verbose = false;
    sub.backend = &backend;
    sub.known_matrix_norm = matrix_norm;
    // The sub-problems are built from the already-scaled lp and the seed lives
    // in that space, so scaling them again would put the warm start in the
    // wrong coordinates.
    sub.scaling.ruiz_iterations = 0;
    sub.scaling.pock_chambolle = false;
    sub.max_iterations = budget;
    // Feasibility sub-problems converge fast, and a sub-solve that cannot look
    // at itself until iteration 40 spends 40 iterations whether or not it
    // arrived at iteration 6. Give a small budget several chances to stop.
    sub.termination_check_frequency =
        std::max<Int>(10, std::min(options.termination_check_frequency, budget / 4));
    sub.time_limit_seconds = std::fmax(0.0, options.time_limit_seconds - elapsed());
    sub.initial_step_size = eta;
    sub.initial_primal_weight = omega;

    // Each sub-solve carries the seed on the side it is solving for and starts
    // the other side at zero, which is trivially feasible for the sub-problem
    // and therefore silent. Carrying the seed's dual into a problem with no
    // objective is what sent the dual objective from 5969 to -11234 the first
    // time this was written.
    const std::vector<double> zero_x(sz(n), 0.0);
    const std::vector<double> zero_y(sz(m), 0.0);

    sub.termination = PdhgOptions::Termination::kPrimalFeasibility;
    sub.warm_x = &seed_x;
    sub.warm_y = &zero_y;
    const PdhgResult primal_polish = solve_pdhg(primal_feas_lp, sub);

    sub.termination = PdhgOptions::Termination::kDualFeasibility;
    sub.warm_x = &zero_x;
    sub.warm_y = &seed_y;
    const PdhgResult dual_polish = solve_pdhg(dual_feas_lp, sub);

    *spent = primal_polish.iterations + dual_polish.iterations;
    result.scaling.scaling.unscale_primal(primal_polish.x, out_x);
    result.scaling.scaling.unscale_dual(dual_polish.y, out_y);
    *out_r = evaluate_residual(original, *out_x, *out_y);
    if (options.verbose) {
      std::printf(
          "  polish  spent %5d (%d primal, %d dual)  ->  primal %.3e  dual %.3e"
          "  gap %.3e\n",
          *spent, (int)primal_polish.iterations, (int)dual_polish.iterations,
          out_r->relative_primal, out_r->relative_dual, out_r->relative_gap);
    }
  };

  auto polish_budget = [&](Int at_iteration) {
    Int budget = static_cast<Int>(options.polish_iteration_factor *
                                  static_cast<double>(at_iteration + 1));
    budget = std::max(budget, options.polish_minimum_iterations);
    budget = std::min(budget, options.polish_maximum_iterations);
    return budget;
  };

  Int iteration = 0;
  Int iterations_since_restart = 0;
  PdhgStatus status = PdhgStatus::kIterationLimit;

  for (; iteration < options.max_iterations; ++iteration) {
    backend.multiply_transpose(lp.kt, d_y.data(), kt_y.data());

    // One PDHG step, retried at a smaller step size if the adaptive rule rejects
    // it. Rejected trials cost one matrix-vector product each.
    for (int attempt = 0;; ++attempt) {
      const double tau = eta / omega;
      const double sigma = eta * omega;

      backend.primal_step(n, tau, d_x.data(), d_c.data(), kt_y.data(),
                          d_lower.data(), d_upper.data(), x_next.data(),
                          dx.data(), x_bar.data());

      backend.multiply(lp.k, x_bar.data(), k_x_bar.data());

      backend.dual_step(m, lp.num_equalities, sigma, d_y.data(), d_q.data(),
                        k_x_bar.data(), k_x.data(), y_next.data(), dy.data(),
                        k_dx.data());

      if (!adaptive) {
        // ||z - T(z)||^2, which the adaptive rule computes as a side effect and
        // the restart tests below need either way.
        movement = backend.weighted_norm_squared(n, m, dx.data(), dy.data(), omega);
        break;
      }

      // PDLP's rule: the step is safe while eta is below the local curvature
      // limit eta_bar, and the new step size may fall fast but grow only slowly.
      // The magnitude matters, not the sign: the step is safe while
      // eta * |dy' K dx| stays under half the weighted movement. Dropping the
      // absolute value here lets eta grow without bound whenever the
      // interaction happens to be negative, and the iterates blow up.
      double interaction = 0.0;
      backend.step_size_terms(n, m, dx.data(), dy.data(), k_dx.data(), omega,
                              &interaction, &movement);
      // movement is ||z - T(z)||^2 in the weighted norm, which is also what the
      // Halpern restart test below needs, so it is hoisted out of this scope.
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

    backend.advance_kx(m, k_x_bar.data(), k_x.data());

    // Reflection: R(z) = T(z) + gamma (T(z) - z). T(z) - z is dx and dy, and
    // K R(z) = K T(z) + gamma K dx with K T(z) sitting in k_x after the line
    // above and K dx in k_dx from the dual step. No product, three updates.
    if (reflection > 0.0) {
      const double g = reflection;
      backend.accumulate(n, g, dx.data(), x_next.data());
      backend.accumulate(m, g, dy.data(), y_next.data());
      backend.accumulate(m, g, k_dx.data(), k_x.data());
    }

    std::swap(d_x, x_next);
    std::swap(d_y, y_next);
    ++iterations_since_restart;

    if (halpern) {
      if (anchor_needs_reset) {
        // z <- T(z), and that point becomes the anchor. This is Algorithm 1's
        // line 6: an epoch begins one plain PDHG step past where the last one
        // ended, and the Halpern counter starts from there.
        backend.copy(d_x.data(), anchor_x.data(), n);
        backend.copy(d_y.data(), anchor_y.data(), m);
        backend.copy(k_x.data(), anchor_kx.data(), m);
        anchor_needs_reset = false;
        halpern_k = 0;
        epoch_start_movement = movement;
      } else {
        const double w = static_cast<double>(halpern_k + 1) /
                         static_cast<double>(halpern_k + 2);
        backend.blend(n, w, d_x.data(), 1.0 - w, anchor_x.data());
        backend.blend(m, w, d_y.data(), 1.0 - w, anchor_y.data());
        // K z follows z by the same combination, since K is linear. Recomputing
        // it instead costs a sparse product per iteration, which on this set
        // was the whole difference between Halpern winning and losing.
        backend.blend(m, w, k_x.data(), 1.0 - w, anchor_kx.data());
        ++halpern_k;
      }
    }

    if (options.restarts && !halpern) {
      backend.accumulate(n, eta, d_x.data(), sum_x.data());
      backend.accumulate(m, eta, d_y.data(), sum_y.data());
      sum_weight += eta;
    }

    // Halpern's own restart test, which is not the KKT one. Equation (10) of
    // Lu and Yang: restart once the fixed point residual ||z - T(z)|| has
    // decayed by a factor of e since the epoch began. That quantity is the
    // `movement` the adaptive step size already computes, so the test is free,
    // and it is checked every iteration rather than on the termination
    // schedule - the whole point is to catch the decay when it happens.
    if (halpern && options.halpern_restart &&
        !anchor_needs_reset &&
        epoch_start_movement > 0.0 && halpern_k >= options.halpern_minimum_epoch) {
      // Comparing squares, so the factor of e becomes e^2.
      if (movement <= epoch_start_movement / 7.389056098930650) {
        anchor_needs_reset = true;
        ++result.restarts;
      }
    }

    const bool check = ((iteration + 1) % options.termination_check_frequency == 0);
    if (!check) continue;

    // Everything from here to the end of the iteration runs on the host, so
    // this is where the data comes back - twice per check, not ten times per
    // iteration.
    d_x.download(&host_x);
    d_y.download(&host_y);
    dy.download(&host_dy);

    if (options.detect_infeasibility &&
        is_infeasibility_certificate(lp, host_dy, options.infeasibility_tolerance,
                                     &infeasibility_work)) {
      status = PdhgStatus::kPrimalInfeasible;
      result.message = "Farkas certificate found in the dual iterate difference";
      ++iteration;
      break;
    }

    // The candidate is whichever of the current iterate and the running average
    // has the smaller KKT error. Both restarting and stopping are judged on it.
    //
    // Only two things read this, and with the defaults neither is on: the
    // average-versus-current comparison needs the averaged scheme, and the
    // restart test needs the KKT rule rather than the fixed-point one. Left
    // unguarded it is two matrix products per termination check spent on a
    // number nothing looks at - and on the GPU those two run on the host, which
    // is where the time was going.
    const bool need_kkt = (options.restarts && !halpern) ||
                          (options.restarts && !options.restart_on_fixed_point) ||
                          options.verbose;
    const double current_kkt =
        need_kkt ? weighted_kkt_error(lp, host_x, host_y, omega, &kkt_work) : 0.0;
    bool use_average = false;
    double candidate_kkt = current_kkt;
    if (options.restarts && !halpern && sum_weight > 0.0) {
      backend.scale_into(n, sum_weight, sum_x.data(), avg_x.data());
      backend.scale_into(m, sum_weight, sum_y.data(), avg_y.data());
      avg_x.download(&host_avg_x);
      avg_y.download(&host_avg_y);
      const double average_kkt =
          weighted_kkt_error(lp, host_avg_x, host_avg_y, omega, &kkt_work);
      if (average_kkt < current_kkt) {
        use_average = true;
        candidate_kkt = average_kkt;
      }
    }
    const std::vector<double>& cand_x = use_average ? host_avg_x : host_x;
    const std::vector<double>& cand_y = use_average ? host_avg_y : host_y;

    // Adopting the candidate has to move it on the device too, not only on the
    // host mirror the checks were computed from.
    auto adopt_candidate = [&]() {
      if (!use_average) return;
      backend.copy(avg_x.data(), d_x.data(), n);
      backend.copy(avg_y.data(), d_y.data(), m);
      host_x = host_avg_x;
      host_y = host_avg_y;
    };

    result.scaling.scaling.unscale_primal(cand_x, &unscaled_x);
    result.scaling.scaling.unscale_dual(cand_y, &unscaled_y);
    const PdhgResidual r = evaluate_residual(original, unscaled_x, unscaled_y);

    if (!std::isfinite(r.primal_residual) || !std::isfinite(r.dual_residual)) {
      status = PdhgStatus::kNumericalError;
      result.message = "iterates stopped being finite";
      ++iteration;
      adopt_candidate();
      break;
    }
    if (passes(r)) {
      status = PdhgStatus::kOptimal;
      ++iteration;
      adopt_candidate();
      break;
    }

    // Polish, once the pair is close enough that the sub-problems start from
    // somewhere useful. Adopted only if the polished pair passes the test the
    // iterate above just failed, so this can end the solve early and cannot end
    // it wrongly.
    if (options.polish_feasibility &&
        options.termination == PdhgOptions::Termination::kFull &&
        iteration + 1 >= next_polish_iteration) {
      // The checkpoints are fixed - 100, 200, 400, 800 and so on - and they
      // advance whether or not this one leads to an attempt; the gap gate
      // below is what decides that. Advancing them only on attempts would
      // cluster attempts at every termination check from the moment the gap
      // first came good, which is the opposite of a geometric schedule.
      while (next_polish_iteration <= iteration + 1) next_polish_iteration *= 2;

      const Int budget = polish_budget(iteration);
      if (r.gap_ok(gap_tolerance) && budget > 0) {
        std::vector<double> try_x;
        std::vector<double> try_y;
        PdhgResidual try_r;
        Int spent = 0;
        run_polish(cand_x, cand_y, budget, &try_x, &try_y, &try_r, &spent);
        ++result.polish_attempts;
        result.polish_iterations += spent;
        if (try_r.converged(tolerance, options.absolute_tolerance,
                            options.require_inf_norm_termination, gap_tolerance)) {
          polished_x = std::move(try_x);
          polished_y = std::move(try_y);
          polished_residual = try_r;
          have_polished = true;
          result.polished = true;
          status = PdhgStatus::kOptimal;
          result.message = "converged after feasibility polishing";
          ++iteration;
          break;
        }
      }
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
      adopt_candidate();
      break;
    }

    if (!options.restarts) continue;


    // cuPDLP's three restart conditions, with its constants. cuPDLPx keeps the
    // shape and changes what is measured: the fixed-point residual
    // ||z - T(z)|| instead of the KKT error. The residual is already computed
    // every iteration for the step size, so measuring it costs nothing, where
    // the KKT error costs two matrix products per check.
    const double residual = std::sqrt(std::fmax(movement, 0.0));
    const bool by_residual = options.restart_on_fixed_point;
    if (by_residual && !std::isfinite(residual_at_restart)) {
      residual_at_restart = residual;
    }
    const double measure = by_residual ? residual : candidate_kkt;
    const double at_restart = by_residual ? residual_at_restart : kkt_at_restart;
    const double previous = by_residual ? last_residual : last_candidate_kkt;

    const bool sufficient = measure <= 0.2 * at_restart;
    const bool necessary = measure <= 0.8 * at_restart && measure > previous;
    const bool too_long =
        static_cast<double>(iterations_since_restart) >= 0.36 * static_cast<double>(iteration + 1);
    last_candidate_kkt = candidate_kkt;
    last_residual = residual;

    if (!(sufficient || necessary || too_long)) continue;

    adopt_candidate();
    backend.multiply(lp.k, d_x.data(), k_x.data());

    if (options.primal_weight_updates && have_previous_restart) {
      // How far each side moved. The smoothing rule measures that over the last
      // two restart periods; the PID one measures it over the current epoch,
      // which is the x^{n,t} - x^{n,0} of the paper.
      const std::vector<double>& base_x =
          options.pid_primal_weight ? restart_x : previous_restart_x;
      const std::vector<double>& base_y =
          options.pid_primal_weight ? restart_y : previous_restart_y;
      double delta_x_sq = 0.0;
      double delta_y_sq = 0.0;
      for (Int j = 0; j < n; ++j) {
        const double d = host_x[sz(j)] - base_x[sz(j)];
        delta_x_sq += d * d;
      }
      for (Int i = 0; i < m; ++i) {
        const double d = host_y[sz(i)] - base_y[sz(i)];
        delta_y_sq += d * d;
      }
      const double delta_x = std::sqrt(delta_x_sq);
      const double delta_y = std::sqrt(delta_y_sq);
      if (delta_x > 1e-12 && delta_y > 1e-12) {
        double candidate = 0.0;
        if (options.pid_primal_weight) {
          // e is the imbalance in log space: positive when the primal moved
          // further than the dual, which is when the weight should come down.
          // With Kp = 0.5 and the other two at zero this is the smoothing rule
          // it replaces, written as a controller.
          const double e = std::log(omega * delta_x / delta_y);
          weight_error_sum += e;
          const double correction = options.primal_weight_kp * e +
                                    options.primal_weight_ki * weight_error_sum +
                                    options.primal_weight_kd *
                                        (e - weight_error_previous);
          weight_error_previous = e;
          candidate = std::exp(std::log(omega) - correction);
        } else {
          // Exponential smoothing in log space, theta = 0.5, so the new weight
          // is the geometric mean of the old one and the ratio of how far the
          // primal and dual moved over the last restart period.
          candidate = std::exp(0.5 * std::log(delta_y / delta_x) +
                               0.5 * std::log(omega));
        }
        if (std::isfinite(candidate) && candidate > 1e-12 && candidate < 1e12) {
          omega = candidate;
        }
      }
    }

    previous_restart_x = restart_x;
    previous_restart_y = restart_y;
    have_previous_restart = true;
    restart_x = host_x;
    restart_y = host_y;

    backend.fill(sum_x.data(), n, 0.0);
    backend.fill(sum_y.data(), m, 0.0);
    sum_weight = 0.0;
    anchor_needs_reset = true;
    iterations_since_restart = 0;
    last_candidate_kkt = kInf;
    last_residual = kInf;
    residual_at_restart = std::sqrt(std::fmax(movement, 0.0));
    // The KKT error is what the other restart rule measures against, and it
    // costs two matrix products, so it is only paid for when it is used.
    if (!options.restart_on_fixed_point) {
      kkt_at_restart = weighted_kkt_error(lp, host_x, host_y, omega, &kkt_work);
    }
    ++result.restarts;
  }

  if (!have_polished) {
    d_x.download(&host_x);
    d_y.download(&host_y);
    result.scaling.scaling.unscale_primal(host_x, &result.x);
    result.scaling.scaling.unscale_dual(host_y, &result.y);
    result.residual = evaluate_residual(original, result.x, result.y);

    // One last polish on the way out. The loop stopped for a reason that was
    // not convergence, so the point it is holding has a real constraint
    // violation in it - and a plan that violates a capacity by 1e-04 is not a
    // plan. Kept only when the polished pair sits closer to the KKT conditions
    // than what it would replace, which is a comparison rather than a
    // judgement.
    const bool worth_trying =
        options.polish_feasibility && options.polish_on_exit &&
        options.termination == PdhgOptions::Termination::kFull &&
        status != PdhgStatus::kOptimal &&
        status != PdhgStatus::kPrimalInfeasible &&
        status != PdhgStatus::kNumericalError &&
        std::isfinite(result.residual.primal_residual) &&
        std::isfinite(result.residual.dual_residual) &&
        elapsed() < options.time_limit_seconds;
    const Int exit_budget = worth_trying ? polish_budget(iteration) : 0;
    if (exit_budget > 0) {
      std::vector<double> try_x;
      std::vector<double> try_y;
      PdhgResidual try_r;
      Int spent = 0;
      run_polish(host_x, host_y, exit_budget, &try_x, &try_y, &try_r, &spent);
      ++result.polish_attempts;
      result.polish_iterations += spent;
      const bool require_inf = options.require_inf_norm_termination;
      // Feasibility is what polishing buys, so feasibility is what has to
      // improve - both sides of it, and strictly. The gap is what it spends,
      // so the gap is capped at whatever the caller already said they would
      // accept, or at what they were about to be handed, whichever is looser.
      // Trading a violated constraint for a wider gap is the intended bargain;
      // trading it for a gap nobody agreed to is not.
      const double gap_ceiling =
          std::fmax(gap_tolerance, result.residual.relative_gap);
      // Neither side may get worse and at least one must get better. Demanding
      // that both improve strictly looks tidier and is wrong: a dual residual
      // of exactly zero is common here, and nothing is strictly less than that,
      // so the test could never pass on the instances that reach it.
      const bool feasibility_improved =
          try_r.relative_primal <= result.residual.relative_primal &&
          try_r.relative_dual <= result.residual.relative_dual &&
          (try_r.relative_primal < result.residual.relative_primal ||
           try_r.relative_dual < result.residual.relative_dual);
      if (feasibility_improved && try_r.relative_gap <= gap_ceiling) {
        result.x = std::move(try_x);
        result.y = std::move(try_y);
        result.residual = try_r;
        result.polished = true;
        if (try_r.converged(tolerance, options.absolute_tolerance, require_inf,
                            gap_tolerance)) {
          status = PdhgStatus::kOptimal;
          result.message = "converged after feasibility polishing";
        } else {
          result.message = result.message.empty()
                               ? "feasibility polished"
                               : result.message + ", feasibility polished";
        }
      }
    }
  } else {
    result.x = std::move(polished_x);
    result.y = std::move(polished_y);
    result.residual = polished_residual;
  }
  result.objective = original.model_objective(result.x);
  // The main loop's count, which is what max_iterations bounds. Polishing is
  // reported next to it rather than folded into it, so that the limit means
  // something a caller can check.
  result.iterations = iteration;
  result.status = status;
  result.final_step_size = eta;
  result.final_primal_weight = omega;
  result.solve_seconds = elapsed();
  return result;
}

}  // namespace sankhya
