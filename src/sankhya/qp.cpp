#include "sankhya/qp.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>

#include "sankhya/backend.hpp"
#include "sankhya/sparse.hpp"

namespace sankhya {
namespace {

// The problem in the form ADMM wants: one matrix A whose rows are the model's
// constraints followed by an identity block for every bounded variable, so the
// whole feasible set is a single box in z.
struct SplitQp {
  SparseMatrix p;
  SparseMatrix a;
  SparseMatrix at;
  std::vector<double> q;
  std::vector<double> l;
  std::vector<double> u;
  Int n = 0;
  Int m = 0;
  double objective_scale = 1.0;
  double objective_offset = 0.0;
};

SplitQp split(const Model& model) {
  SplitQp s;
  s.n = model.num_cols();
  s.objective_scale = (model.sense == ObjSense::kMaximize) ? -1.0 : 1.0;
  s.objective_offset = model.objective_offset;

  s.q.resize(sz(s.n));
  for (Int j = 0; j < s.n; ++j) s.q[sz(j)] = s.objective_scale * model.objective[sz(j)];

  std::vector<Triplet> p_entries;
  p_entries.reserve(sz(model.hessian.nnz()));
  for (Int i = 0; i < model.hessian.rows(); ++i) {
    for (Int k = model.hessian.row_begin(i); k < model.hessian.row_end(i); ++k) {
      p_entries.push_back(Triplet{i, model.hessian.index()[sz(k)],
                                  s.objective_scale * model.hessian.value()[sz(k)]});
    }
  }
  s.p = SparseMatrix::from_triplets(s.n, s.n, std::move(p_entries));

  std::vector<Triplet> a_entries;
  a_entries.reserve(sz(model.constraints.nnz()) + sz(s.n));
  for (Int i = 0; i < model.num_rows(); ++i) {
    for (Int k = model.constraints.row_begin(i); k < model.constraints.row_end(i); ++k) {
      a_entries.push_back(
          Triplet{i, model.constraints.index()[sz(k)], model.constraints.value()[sz(k)]});
    }
  }
  s.l = model.row_lower;
  s.u = model.row_upper;

  Int row = model.num_rows();
  for (Int j = 0; j < s.n; ++j) {
    // Every variable gets a row, even a free one. Uniform structure costs one
    // trivial row and removes a special case from every loop below.
    a_entries.push_back(Triplet{row, j, 1.0});
    s.l.push_back(model.col_lower[sz(j)]);
    s.u.push_back(model.col_upper[sz(j)]);
    ++row;
  }
  s.m = row;
  s.a = SparseMatrix::from_triplets(s.m, s.n, std::move(a_entries));
  s.at = s.a.transpose();
  return s;
}

// Modified Ruiz equilibration, including OSQP's cost-scaling step. Same reason
// as in the linear programming path: without it a first-order method on a badly
// scaled problem converges at a rate that is indistinguishable from not
// converging.
struct QpScaling {
  std::vector<double> d;  // column scaling
  std::vector<double> e;  // row scaling
  double c = 1.0;         // cost scaling
};

QpScaling equilibrate(SplitQp* s, Int iterations) {
  QpScaling scaling;
  scaling.d.assign(sz(s->n), 1.0);
  scaling.e.assign(sz(s->m), 1.0);

  std::vector<double> col_inf(sz(s->n));
  std::vector<double> row_inf(sz(s->m));
  std::vector<double> p_col(sz(s->n));

  for (Int iteration = 0; iteration < iterations; ++iteration) {
    s->a.col_norms(Norm::kInf, col_inf.data());
    s->a.row_norms(Norm::kInf, row_inf.data());
    s->p.col_norms(Norm::kInf, p_col.data());

    std::vector<double> dd(sz(s->n));
    std::vector<double> ee(sz(s->m));
    for (Int j = 0; j < s->n; ++j) {
      const double t = std::fmax(col_inf[sz(j)], p_col[sz(j)]);
      dd[sz(j)] = (t > 0.0) ? 1.0 / std::sqrt(t) : 1.0;
    }
    for (Int i = 0; i < s->m; ++i) {
      ee[sz(i)] = (row_inf[sz(i)] > 0.0) ? 1.0 / std::sqrt(row_inf[sz(i)]) : 1.0;
    }

    s->a.scale_rows(ee.data());
    s->a.scale_cols(dd.data());
    s->p.scale_rows(dd.data());
    s->p.scale_cols(dd.data());
    for (Int j = 0; j < s->n; ++j) {
      s->q[sz(j)] *= dd[sz(j)];
      scaling.d[sz(j)] *= dd[sz(j)];
    }
    for (Int i = 0; i < s->m; ++i) {
      if (s->l[sz(i)] > -kInf) s->l[sz(i)] *= ee[sz(i)];
      if (s->u[sz(i)] < kInf) s->u[sz(i)] *= ee[sz(i)];
      scaling.e[sz(i)] *= ee[sz(i)];
    }

    // Cost scaling: OSQP's addition to plain Ruiz, which stops a very large
    // objective from dominating the whole system.
    s->p.col_norms(Norm::kInf, p_col.data());
    double mean_p = 0.0;
    for (const double v : p_col) mean_p += v;
    mean_p /= std::fmax(1.0, static_cast<double>(s->n));
    double q_inf = 0.0;
    for (const double v : s->q) q_inf = std::fmax(q_inf, std::fabs(v));
    const double gamma = 1.0 / std::fmax(1e-12, std::fmax(mean_p, q_inf));
    if (std::isfinite(gamma) && gamma > 0.0) {
      for (double& v : s->p.value()) v *= gamma;
      for (double& v : s->q) v *= gamma;
      scaling.c *= gamma;
    }
  }
  s->at = s->a.transpose();
  return scaling;
}

}  // namespace

std::string to_string(QpStatus status) {
  switch (status) {
    case QpStatus::kOptimal:
      return "optimal";
    case QpStatus::kIterationLimit:
      return "iteration limit";
    case QpStatus::kTimeLimit:
      return "time limit";
    case QpStatus::kNumericalError:
      return "numerical error";
  }
  return "unknown";
}

QpResult solve_qp(const Model& model, const QpOptions& options) {
  const auto start_time = std::chrono::steady_clock::now();
  auto elapsed = [&start_time]() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time)
        .count();
  };

  const LinAlgBackend& backend = cpu_backend();
  QpResult result;

  SplitQp s = split(model);
  const Int n = s.n;
  const Int m = s.m;
  QpScaling scaling;
  scaling.d.assign(sz(n), 1.0);
  scaling.e.assign(sz(m), 1.0);
  if (options.scaling) scaling = equilibrate(&s, options.ruiz_iterations);

  // Per-constraint step size. Equalities get a much larger value because they
  // are certain to be active at the optimum, which is OSQP's heuristic.
  double rho_bar = options.rho;
  std::vector<double> rho(sz(m));
  auto set_rho = [&]() {
    for (Int i = 0; i < m; ++i) {
      const bool equality = (s.u[sz(i)] - s.l[sz(i)]) < 1e-12;
      rho[sz(i)] = equality ? options.equality_rho_multiplier * rho_bar : rho_bar;
    }
  };
  set_rho();

  std::vector<double> x(sz(n), 0.0);
  std::vector<double> z(sz(m), 0.0);
  std::vector<double> y(sz(m), 0.0);
  std::vector<double> x_tilde(sz(n), 0.0);
  std::vector<double> z_tilde(sz(m), 0.0);
  std::vector<double> rhs(sz(n), 0.0);
  std::vector<double> work_m(sz(m), 0.0);
  std::vector<double> work_n(sz(n), 0.0);
  std::vector<double> ax(sz(m), 0.0);
  std::vector<double> px(sz(n), 0.0);
  std::vector<double> aty(sz(n), 0.0);

  // Conjugate gradient on M = P + sigma I + A' diag(rho) A, which never needs to
  // be formed: one product with P and two with A per iteration.
  std::vector<double> cg_r(sz(n)), cg_p(sz(n)), cg_mp(sz(n)), cg_tmp(sz(m));
  auto apply_m = [&](const std::vector<double>& v, std::vector<double>* out) {
    backend.multiply(s.p, v.data(), out->data());
    backend.multiply(s.a, v.data(), cg_tmp.data());
    for (Int i = 0; i < m; ++i) cg_tmp[sz(i)] *= rho[sz(i)];
    backend.multiply_transpose(s.at, cg_tmp.data(), work_n.data());
    for (Int j = 0; j < n; ++j)
      (*out)[sz(j)] += options.sigma * v[sz(j)] + work_n[sz(j)];
  };

  auto conjugate_gradient = [&](const std::vector<double>& b, std::vector<double>* out,
                                double tolerance) {
    apply_m(*out, &cg_mp);
    for (Int j = 0; j < n; ++j) cg_r[sz(j)] = b[sz(j)] - cg_mp[sz(j)];
    cg_p = cg_r;
    double rr = backend.dot(cg_r.data(), cg_r.data(), n);
    const double threshold = tolerance * tolerance * std::fmax(1e-30, rr);
    Int used = 0;
    for (Int k = 0; k < options.cg_max_iterations && rr > threshold; ++k) {
      apply_m(cg_p, &cg_mp);
      const double denominator = backend.dot(cg_p.data(), cg_mp.data(), n);
      if (!(denominator > 0.0)) break;
      const double alpha = rr / denominator;
      for (Int j = 0; j < n; ++j) {
        (*out)[sz(j)] += alpha * cg_p[sz(j)];
        cg_r[sz(j)] -= alpha * cg_mp[sz(j)];
      }
      const double rr_next = backend.dot(cg_r.data(), cg_r.data(), n);
      const double beta = rr_next / rr;
      for (Int j = 0; j < n; ++j) cg_p[sz(j)] = cg_r[sz(j)] + beta * cg_p[sz(j)];
      rr = rr_next;
      ++used;
    }
    return used;
  };

  auto measure = [&]() {
    QpResidual r;
    backend.multiply(s.a, x.data(), ax.data());
    backend.multiply(s.p, x.data(), px.data());
    backend.multiply_transpose(s.at, y.data(), aty.data());

    // Residuals are judged on the unscaled problem, since that is the one that
    // was asked about. x = D x_scaled, y = c^-1 E y_scaled.
    double primal = 0.0, ax_inf = 0.0, z_inf = 0.0;
    for (Int i = 0; i < m; ++i) {
      const double inv_e = 1.0 / scaling.e[sz(i)];
      const double a_val = inv_e * ax[sz(i)];
      const double z_val = inv_e * z[sz(i)];
      primal = std::fmax(primal, std::fabs(a_val - z_val));
      ax_inf = std::fmax(ax_inf, std::fabs(a_val));
      z_inf = std::fmax(z_inf, std::fabs(z_val));
    }
    double dual = 0.0, px_inf = 0.0, aty_inf = 0.0, q_inf = 0.0;
    for (Int j = 0; j < n; ++j) {
      const double inv_dc = 1.0 / (scaling.d[sz(j)] * scaling.c);
      const double p_val = inv_dc * px[sz(j)];
      const double a_val = inv_dc * aty[sz(j)];
      const double q_val = inv_dc * s.q[sz(j)];
      dual = std::fmax(dual, std::fabs(p_val + q_val + a_val));
      px_inf = std::fmax(px_inf, std::fabs(p_val));
      aty_inf = std::fmax(aty_inf, std::fabs(a_val));
      q_inf = std::fmax(q_inf, std::fabs(q_val));
    }
    r.primal = primal;
    r.dual = dual;
    r.primal_tolerance =
        options.absolute_tolerance + options.relative_tolerance * std::fmax(ax_inf, z_inf);
    r.dual_tolerance = options.absolute_tolerance +
                       options.relative_tolerance *
                           std::fmax(px_inf, std::fmax(aty_inf, q_inf));
    r.absolute_cap = options.max_absolute_residual;
    return r;
  };

  QpStatus status = QpStatus::kIterationLimit;
  Int iteration = 0;
  result.residual = QpResidual{};
  for (; iteration < options.max_iterations; ++iteration) {
    // rhs = sigma x - q + A'(rho z - y)
    for (Int i = 0; i < m; ++i) work_m[sz(i)] = rho[sz(i)] * z[sz(i)] - y[sz(i)];
    backend.multiply_transpose(s.at, work_m.data(), rhs.data());
    for (Int j = 0; j < n; ++j)
      rhs[sz(j)] += options.sigma * x[sz(j)] - s.q[sz(j)];

    x_tilde = x;  // warm start the inner solve from the current iterate
    result.cg_iterations += conjugate_gradient(rhs, &x_tilde, options.cg_tolerance);

    // Eliminating nu from the KKT system leaves z_tilde = A x_tilde exactly.
    backend.multiply(s.a, x_tilde.data(), z_tilde.data());

    for (Int j = 0; j < n; ++j)
      x[sz(j)] = options.alpha * x_tilde[sz(j)] + (1.0 - options.alpha) * x[sz(j)];

    for (Int i = 0; i < m; ++i) {
      const std::size_t si = sz(i);
      const double relaxed = options.alpha * z_tilde[si] + (1.0 - options.alpha) * z[si];
      const double candidate = relaxed + y[si] / rho[si];
      double z_new = candidate;
      if (z_new < s.l[si]) z_new = s.l[si];
      if (z_new > s.u[si]) z_new = s.u[si];
      y[si] += rho[si] * (relaxed - z_new);
      z[si] = z_new;
    }

    if ((iteration + 1) % options.termination_check_frequency != 0) continue;

    const QpResidual r = measure();
    if (!std::isfinite(r.primal) || !std::isfinite(r.dual)) {
      status = QpStatus::kNumericalError;
      result.message = "iterates stopped being finite";
      ++iteration;
      break;
    }
    if (r.converged()) {
      status = QpStatus::kOptimal;
      result.residual = r;
      ++iteration;
      break;
    }
    if (options.verbose && (iteration + 1) % options.log_frequency == 0) {
      std::printf("  iter %6d  primal %.3e / %.3e   dual %.3e / %.3e   rho %.3e\n",
                  iteration + 1, r.primal, r.primal_tolerance, r.dual,
                  r.dual_tolerance, rho_bar);
    }
    if (elapsed() > options.time_limit_seconds) {
      status = QpStatus::kTimeLimit;
      ++iteration;
      break;
    }

    if (options.adaptive_rho) {
      // OSQP's rule: move rho by the square root of the ratio of the two
      // relatively-scaled residuals, and only when the change is worth it.
      const double num = r.primal / std::fmax(1e-12, r.primal_tolerance);
      const double den = r.dual / std::fmax(1e-12, r.dual_tolerance);
      const double factor = std::sqrt(num / std::fmax(1e-12, den));
      if (std::isfinite(factor) &&
          (factor > options.rho_update_threshold ||
           factor < 1.0 / options.rho_update_threshold)) {
        rho_bar = std::fmin(1e6, std::fmax(1e-6, rho_bar * factor));
        set_rho();
        result.rho_updates++;
      }
    }
  }

  // Polishing. Guess the active set from the signs of the duals, then solve the
  // equality-constrained problem that guess implies.
  //
  // OSQP does this with a direct factorisation of the reduced KKT system. That
  // system is indefinite, so conjugate gradient cannot touch it - but
  // eliminating the duals of the active rows turns it into
  //
  //   (P + delta I + (A_act' A_act)/delta) x = -q + (A_act' b_act)/delta
  //
  // which is symmetric positive definite, and therefore solvable with exactly
  // the machinery already here.
  if (options.polish && status == QpStatus::kOptimal) {
    std::vector<Int> active;
    std::vector<double> target;
    for (Int i = 0; i < m; ++i) {
      if (y[sz(i)] < -1e-8) {
        active.push_back(i);
        target.push_back(s.l[sz(i)]);
      } else if (y[sz(i)] > 1e-8) {
        active.push_back(i);
        target.push_back(s.u[sz(i)]);
      }
    }

    const double delta = options.polish_regularisation;
    std::vector<double> row_active(sz(m), 0.0);
    std::vector<double> row_target(sz(m), 0.0);
    for (std::size_t idx = 0; idx < active.size(); ++idx) {
      row_active[sz(active[idx])] = 1.0;
      row_target[sz(active[idx])] = target[idx];
    }

    auto apply_polish = [&](const std::vector<double>& v, std::vector<double>* out) {
      backend.multiply(s.p, v.data(), out->data());
      backend.multiply(s.a, v.data(), cg_tmp.data());
      for (Int i = 0; i < m; ++i) cg_tmp[sz(i)] *= row_active[sz(i)] / delta;
      backend.multiply_transpose(s.at, cg_tmp.data(), work_n.data());
      for (Int j = 0; j < n; ++j)
        (*out)[sz(j)] += delta * v[sz(j)] + work_n[sz(j)];
    };

    std::vector<double> rhs_polish(sz(n), 0.0);
    for (Int i = 0; i < m; ++i) cg_tmp[sz(i)] = row_target[sz(i)] / delta;
    backend.multiply_transpose(s.at, cg_tmp.data(), rhs_polish.data());
    for (Int j = 0; j < n; ++j) rhs_polish[sz(j)] -= s.q[sz(j)];

    std::vector<double> polished = x;
    std::vector<double> pr(sz(n)), pp(sz(n)), pmp(sz(n));
    // Iterative refinement around the regularised solve, as OSQP does: the
    // regularisation is what makes the system solvable and also what makes the
    // answer slightly wrong, and refinement takes that back out.
    for (Int pass = 0; pass < options.polish_refinement_steps; ++pass) {
      apply_polish(polished, &pmp);
      for (Int j = 0; j < n; ++j) pr[sz(j)] = rhs_polish[sz(j)] - pmp[sz(j)];
      pp = pr;
      double rr = backend.dot(pr.data(), pr.data(), n);
      if (rr < 1e-30) break;
      const double stop = 1e-14 * rr;
      for (Int k = 0; k < options.polish_cg_iterations && rr > stop; ++k) {
        apply_polish(pp, &pmp);
        const double den = backend.dot(pp.data(), pmp.data(), n);
        if (!(den > 0.0)) break;
        const double alpha = rr / den;
        for (Int j = 0; j < n; ++j) {
          polished[sz(j)] += alpha * pp[sz(j)];
          pr[sz(j)] -= alpha * pmp[sz(j)];
        }
        const double rr_next = backend.dot(pr.data(), pr.data(), n);
        const double beta = rr_next / rr;
        for (Int j = 0; j < n; ++j) pp[sz(j)] = pr[sz(j)] + beta * pp[sz(j)];
        rr = rr_next;
      }
    }

    // The duals have to be recovered too. Eliminating them is what made the
    // system positive definite, so they come back from the same relation:
    // y_active = (A_active x - target) / delta, and zero everywhere else.
    // A first version of this kept the old duals and only replaced x, which
    // made the dual residual worse by construction and meant the polished point
    // was rejected every single time - fifteen instances out of fifteen.
    const std::vector<double> saved_x = x;
    const std::vector<double> saved_z = z;
    const std::vector<double> saved_y = y;
    x = polished;
    backend.multiply(s.a, x.data(), ax.data());
    for (Int i = 0; i < m; ++i) {
      const std::size_t si = sz(i);
      if (row_active[si] > 0.0) {
        y[si] = (ax[si] - row_target[si]) / delta;
        z[si] = row_target[si];
      } else {
        y[si] = 0.0;
        z[si] = std::fmin(std::fmax(ax[si], s.l[si]), s.u[si]);
      }
    }
    const QpResidual after = measure();
    const bool better = std::isfinite(after.primal) && std::isfinite(after.dual) &&
                        after.primal <= result.residual.primal + 1e-12 &&
                        after.dual <= std::fmax(result.residual.dual, 1e-12);
    if (options.verbose) {
      std::printf(
          "  polish: %zu of %d rows active   primal %.3e -> %.3e   "
          "dual %.3e -> %.3e   %s\n",
          active.size(), static_cast<int>(m), result.residual.primal, after.primal,
          result.residual.dual, after.dual, better ? "accepted" : "rejected");
    }
    if (better) {
      result.polished = true;
    } else {
      x = saved_x;
      z = saved_z;
      y = saved_y;
    }
  }

  // Unscale: x = D x_scaled, y = c^-1 E y_scaled.
  result.x.resize(sz(n));
  for (Int j = 0; j < n; ++j) result.x[sz(j)] = scaling.d[sz(j)] * x[sz(j)];
  result.y.resize(sz(m));
  for (Int i = 0; i < m; ++i)
    result.y[sz(i)] = scaling.e[sz(i)] * y[sz(i)] / scaling.c;

  double objective = model.objective_offset;
  for (Int j = 0; j < n; ++j) objective += model.objective[sz(j)] * result.x[sz(j)];
  if (model.has_hessian()) {
    std::vector<double> hx(sz(n));
    model.hessian.multiply(result.x.data(), hx.data());
    objective += 0.5 * backend.dot(result.x.data(), hx.data(), n);
  }

  result.objective = objective;
  result.iterations = iteration;
  result.status = status;
  result.residual = measure();
  result.final_rho = rho_bar;
  result.solve_seconds = elapsed();
  return result;
}

}  // namespace sankhya
