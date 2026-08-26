#include "sankhya/qp.hpp"

#include "sankhya/ldl.hpp"

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
  // The KKT system, as a lower triangle stored row-wise:
  //
  //     [ P + sigma I      A'        ]      rows 0..n-1 hold P's lower half
  //     [ A            -diag(1/rho)  ]      rows n..n+m-1 hold A and -1/rho
  //
  // Only the values on the trailing diagonal change when rho does, so the
  // pattern is analysed once and the numbers are refactorised as needed.
  LdlFactor kkt;
  std::vector<double> kkt_rhs(sz(n + m), 0.0);
  bool kkt_ready = false;
  auto build_kkt = [&]() {
    std::vector<Triplet> entries;
    entries.reserve(sz(s.p.nnz() + s.a.nnz() + n + m));
    for (Int j = 0; j < n; ++j) {
      for (Int e = s.p.row_begin(j); e < s.p.row_end(j); ++e) {
        const Int c = s.p.index()[sz(e)];
        if (c <= j) entries.push_back(Triplet{j, c, s.p.value()[sz(e)]});
      }
      entries.push_back(Triplet{j, j, options.sigma});
    }
    for (Int i = 0; i < m; ++i) {
      for (Int e = s.a.row_begin(i); e < s.a.row_end(i); ++e)
        entries.push_back(Triplet{n + i, s.a.index()[sz(e)], s.a.value()[sz(e)]});
      entries.push_back(Triplet{n + i, n + i, -1.0 / rho[sz(i)]});
    }
    return SparseMatrix::from_triplets(n + m, n + m, std::move(entries));
  };
  bool use_direct = options.direct;
  auto refactorize_kkt = [&](std::string* error) {
    const SparseMatrix k = build_kkt();
    if (!kkt_ready) {
      const Int budget =
          static_cast<Int>(options.max_fill_ratio * static_cast<double>(k.nnz()));
      if (!kkt.analyse(k, budget, error)) {
        // Decided from the pattern, before any numbers are touched, and the
        // counting stops as soon as the answer is known.
        result.kkt_fill_ratio = k.nnz() > 0
            ? static_cast<double>(kkt.predicted_nonzeros()) / k.nnz() : 0.0;
        use_direct = false;
        result.fell_back_to_cg = true;
        if (error) error->clear();
        return true;
      }
      result.kkt_fill_ratio = k.nnz() > 0
          ? static_cast<double>(kkt.predicted_nonzeros()) / k.nnz() : 0.0;
      kkt_ready = true;
    }
    return kkt.factorize(k, LdlOptions{}, error);
  };

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
  std::string kkt_error;
  if (use_direct && !refactorize_kkt(&kkt_error)) {
    result.status = QpStatus::kNumericalError;
    result.message = "could not factorise the KKT system: " + kkt_error;
    return result;
  }

  result.residual = QpResidual{};
  for (; iteration < options.max_iterations; ++iteration) {
    // rhs = sigma x - q + A'(rho z - y)
    for (Int i = 0; i < m; ++i) work_m[sz(i)] = rho[sz(i)] * z[sz(i)] - y[sz(i)];
    backend.multiply_transpose(s.at, work_m.data(), rhs.data());
    for (Int j = 0; j < n; ++j)
      rhs[sz(j)] += options.sigma * x[sz(j)] - s.q[sz(j)];

    if (use_direct) {
      // [ P + sigma I   A'      ] [x~]   [ sigma x - q   ]
      // [ A          -1/rho     ] [nu]   [ z - y/rho     ]
      //
      // The top block is sigma x - q on its own. `rhs` above is the right hand
      // side of the *reduced* system, which has already had A'(rho z - y)
      // folded into it - that term is what eliminating nu produces, and putting
      // it back into a system that still has nu in it counts it twice.
      for (Int j = 0; j < n; ++j)
        kkt_rhs[sz(j)] = options.sigma * x[sz(j)] - s.q[sz(j)];
      for (Int i = 0; i < m; ++i)
        kkt_rhs[sz(n + i)] = z[sz(i)] - y[sz(i)] / rho[sz(i)];
      kkt.solve(&kkt_rhs);
      for (Int j = 0; j < n; ++j) x_tilde[sz(j)] = kkt_rhs[sz(j)];
      // nu came back in the tail, and z~ = z + (nu - y)/rho.
      for (Int i = 0; i < m; ++i)
        z_tilde[sz(i)] = z[sz(i)] + (kkt_rhs[sz(n + i)] - y[sz(i)]) / rho[sz(i)];
    } else {
      x_tilde = x;  // warm start the inner solve from the current iterate
      result.cg_iterations += conjugate_gradient(rhs, &x_tilde, options.cg_tolerance);

      // Eliminating nu from the KKT system leaves z_tilde = A x_tilde exactly.
      backend.multiply(s.a, x_tilde.data(), z_tilde.data());
    }

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
        // Only the trailing diagonal moved, so the pattern still holds and this
        // is a numeric factorisation on the ordering already chosen.
        if (use_direct && !refactorize_kkt(&kkt_error)) {
          result.status = QpStatus::kNumericalError;
          result.message = "could not refactorise after a rho update: " + kkt_error;
          return result;
        }
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
    std::vector<double> polish_nu(sz(m), 0.0);
    std::vector<double> row_active(sz(m), 0.0);
    std::vector<double> row_target(sz(m), 0.0);
    for (std::size_t idx = 0; idx < active.size(); ++idx) {
      row_active[sz(active[idx])] = 1.0;
      row_target[sz(active[idx])] = target[idx];
    }

    // The reduced KKT system, solved rather than reduced further:
    //
    //     [ P + delta I     A_act'   ] [x]   [ -q      ]
    //     [ A_act         -delta I   ] [nu] = [ b_act   ]
    //
    // Eliminating nu from this is what the previous version did, and it turns a
    // quasi-definite system into a positive definite one with a condition
    // number of order 1/delta - which is why conjugate gradient never got
    // anywhere and polishing was rejected fifteen times out of fifteen. Left
    // alone the system is quasi-definite and factorises, and the duals come
    // straight out of it instead of being recovered by dividing by delta.
    const Int k = static_cast<Int>(active.size());
    std::vector<Triplet> polish_entries;
    for (Int j = 0; j < n; ++j) {
      for (Int e = s.p.row_begin(j); e < s.p.row_end(j); ++e) {
        const Int c = s.p.index()[sz(e)];
        if (c <= j) polish_entries.push_back(Triplet{j, c, s.p.value()[sz(e)]});
      }
      polish_entries.push_back(Triplet{j, j, delta});
    }
    for (Int idx = 0; idx < k; ++idx) {
      const Int row = active[sz(idx)];
      for (Int e = s.a.row_begin(row); e < s.a.row_end(row); ++e)
        polish_entries.push_back(
            Triplet{n + idx, s.a.index()[sz(e)], s.a.value()[sz(e)]});
      polish_entries.push_back(Triplet{n + idx, n + idx, -delta});
    }
    const SparseMatrix polish_kkt =
        SparseMatrix::from_triplets(n + k, n + k, std::move(polish_entries));

    LdlFactor polish_factor;
    std::string polish_error;
    std::vector<double> polished = x;
    bool polish_ok = false;
    if (polish_factor.analyse(polish_kkt,
                              static_cast<Int>(options.max_fill_ratio *
                                               static_cast<double>(polish_kkt.nnz())),
                              &polish_error) &&
        polish_factor.factorize(polish_kkt, LdlOptions{}, &polish_error)) {
      std::vector<double> rhs_polish(sz(n + k), 0.0);
      for (Int j = 0; j < n; ++j) rhs_polish[sz(j)] = -s.q[sz(j)];
      for (Int idx = 0; idx < k; ++idx) rhs_polish[sz(n + idx)] = target[sz(idx)];

      std::vector<double> t = rhs_polish;
      polish_factor.solve(&t);

      // Iterative refinement against the system without the regularisation.
      // The delta is what makes it solvable and also what makes the answer
      // slightly wrong; this takes that back out.
      std::vector<double> residual(sz(n + k), 0.0);
      std::vector<double> px_full(sz(n), 0.0);
      for (Int pass = 0; pass < options.polish_refinement_steps; ++pass) {
        for (Int j = 0; j < n; ++j) work_n[sz(j)] = t[sz(j)];
        backend.multiply(s.p, work_n.data(), px_full.data());
        for (Int j = 0; j < n; ++j) residual[sz(j)] = -s.q[sz(j)] - px_full[sz(j)];
        for (Int idx = 0; idx < k; ++idx) {
          const Int row = active[sz(idx)];
          double dot = 0.0;
          for (Int e = s.a.row_begin(row); e < s.a.row_end(row); ++e) {
            const Int c = s.a.index()[sz(e)];
            dot += s.a.value()[sz(e)] * t[sz(c)];
            residual[sz(c)] -= s.a.value()[sz(e)] * t[sz(n + idx)];
          }
          residual[sz(n + idx)] = target[sz(idx)] - dot;
        }
        polish_factor.solve(&residual);
        for (Int i = 0; i < n + k; ++i) t[sz(i)] += residual[sz(i)];
      }

      for (Int j = 0; j < n; ++j) polished[sz(j)] = t[sz(j)];
      for (Int idx = 0; idx < k; ++idx)
        row_target[sz(active[sz(idx)])] = target[sz(idx)];
      polish_nu.assign(sz(m), 0.0);
      for (Int idx = 0; idx < k; ++idx)
        polish_nu[sz(active[sz(idx)])] = t[sz(n + idx)];
      polish_ok = true;
    }
    if (!polish_ok) {
      if (options.verbose)
        std::printf("  polish: skipped, %s\n", polish_error.c_str());
    } else {
    // was rejected every single time - fifteen instances out of fifteen.
    const std::vector<double> saved_x = x;
    const std::vector<double> saved_z = z;
    const std::vector<double> saved_y = y;
    x = polished;
    backend.multiply(s.a, x.data(), ax.data());
    for (Int i = 0; i < m; ++i) {
      const std::size_t si = sz(i);
      if (row_active[si] > 0.0) {
        y[si] = polish_nu[si];  // straight out of the solve
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
