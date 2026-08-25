#include "sankhya/simplex.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>

namespace sankhya {

LogicalForm to_logical_form(const StandardLp& lp) {
  LogicalForm form;
  form.num_structural = lp.num_cols();
  form.num_rows = lp.num_rows();
  form.num_equalities = lp.num_equalities;
  form.objective_scale = lp.objective_scale;
  form.objective_offset = lp.objective_offset;

  // Every row gets a logical, including the equalities. An equality's logical is
  // fixed at zero, so it changes nothing about the feasible set, and in exchange
  // the all-logical basis is the identity and is nonsingular by construction.
  //
  // The first version only gave logicals to inequality rows and picked a
  // structural column for each equality by a greedy rule. That is how real
  // crash bases work, but a naive version of it produced singular starting
  // bases: scfxm1, bandm and degen2 all failed to factorise before the first
  // iteration. Starting from the identity and letting phase one do the work is
  // slower to a first feasible point and cannot fail.
  const Int columns = form.num_structural + lp.num_rows();

  form.cost.assign(sz(columns), 0.0);
  for (Int j = 0; j < form.num_structural; ++j) form.cost[sz(j)] = lp.c[sz(j)];
  form.rhs = lp.q;
  form.lower.assign(sz(columns), 0.0);
  form.upper.assign(sz(columns), kInf);
  for (Int j = 0; j < form.num_structural; ++j) {
    form.lower[sz(j)] = lp.lower[sz(j)];
    form.upper[sz(j)] = lp.upper[sz(j)];
  }
  for (Int i = 0; i < lp.num_rows(); ++i) {
    const Int column = form.num_structural + i;
    form.lower[sz(column)] = 0.0;
    // a'x = q becomes a'x - s = q with s fixed at zero;
    // a'x >= q becomes a'x - s = q with s >= 0.
    form.upper[sz(column)] = (i < lp.num_equalities) ? 0.0 : kInf;
  }

  std::vector<Triplet> entries;
  entries.reserve(sz(lp.k.nnz()) + sz(lp.num_rows()));
  for (Int j = 0; j < form.num_structural; ++j) {
    for (Int e = lp.kt.row_begin(j); e < lp.kt.row_end(j); ++e) {
      entries.push_back(Triplet{j, lp.kt.index()[sz(e)], lp.kt.value()[sz(e)]});
    }
  }
  for (Int i = 0; i < lp.num_rows(); ++i) {
    entries.push_back(Triplet{form.num_structural + i, i, -1.0});
  }
  form.columns = SparseMatrix::from_triplets(columns, lp.num_rows(), std::move(entries));
  form.rows = form.columns.transpose();
  return form;
}

bool SimplexBasis::set_initial(const LogicalForm& form, std::string* error) {
  const Int m = form.num_rows;
  const Int columns = form.columns.rows();
  basic_.assign(sz(m), -1);
  status_.assign(sz(columns), VarStatus::kAtLower);

  for (Int j = 0; j < columns; ++j) {
    if (form.lower[sz(j)] <= -kInf && form.upper[sz(j)] >= kInf) {
      status_[sz(j)] = VarStatus::kFree;
    } else if (form.lower[sz(j)] <= -kInf) {
      status_[sz(j)] = VarStatus::kAtUpper;
    }
  }

  // The all-logical basis: -I, nonsingular whatever the model looks like.
  for (Int i = 0; i < m; ++i) {
    const Int logical = form.num_structural + i;
    basic_[sz(i)] = logical;
    status_[sz(logical)] = VarStatus::kBasic;
  }
  return refactorize(form, error);
}

bool SimplexBasis::refactorize(const LogicalForm& form, std::string* error) {
  // The updates are only discarded once there is a factorisation to replace
  // them with. A failed refactorisation leaves the basis exactly as it was, so
  // the caller still has something to roll back.
  if (!lu_.factorize(form.columns, basic_, LuOptions{}, error)) return false;
  updates_ = 0;
  growth_ = 1.0;
  updates_list_.clear();
  return true;
}

bool SimplexBasis::rollback_last_update() {
  if (updates_list_.empty()) return false;
  const Update& u = updates_list_.back();
  if (u.entered < 0 || u.left < 0) return false;
  basic_[sz(u.row)] = u.left;
  status_[sz(u.left)] = VarStatus::kBasic;
  status_[sz(u.entered)] = u.entered_was;
  updates_list_.pop_back();
  if (updates_ > 0) --updates_;
  return true;
}

// B^-1 = E_k^-1 ... E_1^-1 B_0^-1, so the factorisation goes first and the
// updates follow in the order they were made.
void SimplexBasis::ftran(std::vector<double>* x) const {
  lu_.ftran(x);
  for (const Update& u : updates_list_) {
    const double at_pivot = (*x)[sz(u.row)];
    if (at_pivot == 0.0) continue;
    const double scaled = at_pivot / u.pivot;
    for (std::size_t k = 0; k < u.rows.size(); ++k)
      (*x)[sz(u.rows[k])] -= u.values[k] * scaled;
    (*x)[sz(u.row)] = scaled;
  }
}

// B^-T = B_0^-T (E_1^-1)^T ... (E_k^-1)^T, which reads right to left: the
// updates in reverse, then the factorisation.
void SimplexBasis::btran(std::vector<double>* x) const {
  for (std::size_t t = updates_list_.size(); t-- > 0;) {
    const Update& u = updates_list_[t];
    double sum = (*x)[sz(u.row)];
    for (std::size_t k = 0; k < u.rows.size(); ++k)
      sum -= u.values[k] * (*x)[sz(u.rows[k])];
    (*x)[sz(u.row)] = sum / u.pivot;
  }
  lu_.btran(x);
}

void SimplexBasis::compute_primal(const LogicalForm& form,
                                  std::vector<double>* values) const {
  const Int m = form.num_rows;
  const Int columns = form.columns.rows();
  values->assign(sz(columns), 0.0);

  // Non-basic variables sit on a bound.
  for (Int j = 0; j < columns; ++j) {
    switch (status_[sz(j)]) {
      case VarStatus::kAtLower:
        (*values)[sz(j)] = std::isfinite(form.lower[sz(j)]) ? form.lower[sz(j)] : 0.0;
        break;
      case VarStatus::kAtUpper:
        (*values)[sz(j)] = std::isfinite(form.upper[sz(j)]) ? form.upper[sz(j)] : 0.0;
        break;
      case VarStatus::kFree:
        (*values)[sz(j)] = 0.0;
        break;
      case VarStatus::kBasic:
        break;
    }
  }

  // rhs minus the contribution of everything non-basic.
  std::vector<double> residual = form.rhs;
  for (Int j = 0; j < columns; ++j) {
    if (status_[sz(j)] == VarStatus::kBasic) continue;
    const double value = (*values)[sz(j)];
    if (value == 0.0) continue;
    for (Int e = form.columns.row_begin(j); e < form.columns.row_end(j); ++e) {
      residual[sz(form.columns.index()[sz(e)])] -= form.columns.value()[sz(e)] * value;
    }
  }

  ftran(&residual);
  for (Int i = 0; i < m; ++i) (*values)[sz(basic_[sz(i)])] = residual[sz(i)];
}

void SimplexBasis::compute_duals(const LogicalForm& form, std::vector<double>* duals,
                                 std::vector<double>* reduced_costs) const {
  const Int m = form.num_rows;
  const Int columns = form.columns.rows();

  std::vector<double> cb(sz(m), 0.0);
  for (Int i = 0; i < m; ++i) cb[sz(i)] = form.cost[sz(basic_[sz(i)])];
  btran(&cb);
  *duals = cb;

  reduced_costs->assign(sz(columns), 0.0);
  for (Int j = 0; j < columns; ++j) {
    if (status_[sz(j)] == VarStatus::kBasic) continue;
    double dot = 0.0;
    for (Int e = form.columns.row_begin(j); e < form.columns.row_end(j); ++e) {
      dot += form.columns.value()[sz(e)] * cb[sz(form.columns.index()[sz(e)])];
    }
    (*reduced_costs)[sz(j)] = form.cost[sz(j)] - dot;
  }
}

void SimplexBasis::pivot_row(const LogicalForm& form, Int row,
                             std::vector<double>* rho) const {
  rho->assign(sz(form.num_rows), 0.0);
  if (row < 0 || row >= form.num_rows) return;
  (*rho)[sz(row)] = 1.0;
  btran(rho);
}

void SimplexBasis::ftran_column(const LogicalForm& form, Int column,
                                std::vector<double>* out) const {
  out->assign(sz(form.num_rows), 0.0);
  for (Int e = form.columns.row_begin(column); e < form.columns.row_end(column); ++e) {
    (*out)[sz(form.columns.index()[sz(e)])] = form.columns.value()[sz(e)];
  }
  ftran(out);
}

bool SimplexBasis::pivot(const LogicalForm& form, Int leaving_row, Int entering,
                         VarStatus leaving_to,
                         const std::vector<double>& pivotal_column,
                         std::string* error) {
  // Dividing by this is the whole update, so refusing here and refactorising is
  // cheaper than carrying a factor that is mostly rounding error.
  constexpr double kMinimumPivot = 1e-11;
  // Entries this far below the pivot cannot change an answer at double
  // precision, and keeping them would make every update denser than the one
  // before it.
  constexpr double kDropTolerance = 1e-13;

  if (leaving_row < 0 || leaving_row >= form.num_rows) {
    if (error) *error = "pivot row out of range";
    return false;
  }
  const double pivot_value = pivotal_column[sz(leaving_row)];
  if (!(std::fabs(pivot_value) > kMinimumPivot)) {
    if (error) {
      *error = "pivot element " + std::to_string(pivot_value) +
               " is too small to update the basis with";
    }
    return false;
  }

  Update update;
  update.row = leaving_row;
  update.pivot = pivot_value;
  const double drop = kDropTolerance * std::fabs(pivot_value);
  for (Int i = 0; i < form.num_rows; ++i) {
    if (i == leaving_row) continue;
    const double v = pivotal_column[sz(i)];
    if (std::fabs(v) <= drop) continue;
    update.rows.push_back(i);
    update.values.push_back(v);
  }

  const Int leaving = basic_[sz(leaving_row)];
  update.entered = entering;
  update.left = leaving;
  update.entered_was = status_[sz(entering)];
  basic_[sz(leaving_row)] = entering;
  status_[sz(entering)] = VarStatus::kBasic;
  status_[sz(leaving)] = leaving_to;

  growth_ = std::fmax(growth_, 1.0 / std::fabs(pivot_value));
  updates_list_.push_back(std::move(update));
  ++updates_;
  return true;
}

bool SimplexBasis::pivot(const LogicalForm& form, Int leaving_row, Int entering,
                         VarStatus leaving_to, std::string* error) {
  std::vector<double> column;
  ftran_column(form, entering, &column);
  return pivot(form, leaving_row, entering, leaving_to, column, error);
}

}  // namespace sankhya

namespace sankhya {
namespace {

// Phase one gives each basic variable a cost that points it back inside its
// bounds: +1 if it is above its upper bound, -1 if below its lower. Minimising
// that sum is minimising total infeasibility, and it reaches zero exactly when
// the basis is primal feasible. No artificial columns and no big constant, so
// the matrix never changes between phases.
double infeasibility_cost(double value, double lower, double upper,
                          double tolerance) {
  if (value < lower - tolerance) return -1.0;
  if (value > upper + tolerance) return 1.0;
  return 0.0;
}

double total_infeasibility(const LogicalForm& form, const SimplexBasis& basis,
                           const std::vector<double>& z, double tolerance) {
  double total = 0.0;
  for (Int i = 0; i < form.num_rows; ++i) {
    const Int j = basis.basic()[sz(i)];
    const double value = z[sz(j)];
    if (value < form.lower[sz(j)] - tolerance) total += form.lower[sz(j)] - value;
    if (value > form.upper[sz(j)] + tolerance) total += value - form.upper[sz(j)];
  }
  return total;
}

}  // namespace

std::string to_string(SimplexStatus status) {
  switch (status) {
    case SimplexStatus::kOptimal: return "optimal";
    case SimplexStatus::kUnbounded: return "unbounded";
    case SimplexStatus::kInfeasible: return "infeasible";
    case SimplexStatus::kIterationLimit: return "iteration limit";
    case SimplexStatus::kTimeLimit: return "time limit";
    case SimplexStatus::kNumericalError: return "numerical error";
  }
  return "unknown";
}

SimplexResult solve_simplex(const StandardLp& lp, const SimplexOptions& options) {
  const auto start_time = std::chrono::steady_clock::now();
  auto elapsed = [&start_time]() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time)
        .count();
  };

  SimplexResult result;
  const LogicalForm form = to_logical_form(lp);
  const Int m = form.num_rows;
  const Int columns = form.columns.rows();

  SimplexBasis basis;
  std::string error;
  if (!basis.set_initial(form, &error)) {
    result.message = error;
    return result;
  }
  result.refactorizations = 1;

  std::vector<double> z;
  std::vector<double> duals;
  std::vector<double> reduced;
  std::vector<double> column;
  std::vector<double> phase_cost(sz(columns), 0.0);

  bool phase_one = true;
  Int iteration = 0;
  SimplexStatus status = SimplexStatus::kIterationLimit;

  // Two ratios counting as equal, for the tie-break Bland's rule needs.
  constexpr double kRatioTie = 1e-9;
  Int degenerate_run = 0;

  // Devex reference weights. One per column; the reference framework is the
  // nonbasic set at the moment they were last all set to one.
  std::vector<double> weight(sz(columns), 1.0);
  std::vector<double> pivot_row_work;
  std::vector<double> rho;
  bool bland = false;

  for (; iteration < options.max_iterations; ++iteration) {
    if (elapsed() > options.time_limit_seconds) {
      status = SimplexStatus::kTimeLimit;
      break;
    }
    result.worst_update_growth =
        std::fmax(result.worst_update_growth, basis.update_growth());
    if (basis.updates_since_refactorization() >= options.refactorization_frequency) {
      // If it will not factorise, walk back one pivot at a time until it does.
      // Each rollback throws away an iteration's work and no more, which is
      // cheap next to the alternative of declaring a numerical failure on a
      // model that a factorisation from one pivot earlier handles fine.
      Int rolled = 0;
      while (!basis.refactorize(form, &error)) {
        if (rolled >= options.max_rollback || !basis.rollback_last_update()) {
          status = SimplexStatus::kNumericalError;
          result.message = "basis will not factorise after rolling back " +
                           std::to_string(rolled) + " pivots: " + error;
          break;
        }
        ++rolled;
        ++result.rollbacks;
      }
      if (status == SimplexStatus::kNumericalError) break;
      result.refactorizations++;
    }

    basis.compute_primal(form, &z);

    if (phase_one) {
      const double infeasibility =
          total_infeasibility(form, basis, z, options.primal_tolerance);
      if (infeasibility <= options.primal_tolerance) {
        phase_one = false;
        result.phase_one_iterations = iteration;
      }
    }

    // The cost this iteration optimises: the real objective in phase two, the
    // infeasibility direction in phase one.
    std::fill(phase_cost.begin(), phase_cost.end(), 0.0);
    if (phase_one) {
      for (Int i = 0; i < m; ++i) {
        const Int j = basis.basic()[sz(i)];
        phase_cost[sz(j)] = infeasibility_cost(z[sz(j)], form.lower[sz(j)],
                                               form.upper[sz(j)],
                                               options.primal_tolerance);
      }
    } else {
      phase_cost = form.cost;
    }

    LogicalForm priced = form;
    priced.cost = phase_cost;
    basis.compute_duals(priced, &duals, &reduced);

    // Dantzig pricing: the most favourable reduced cost. Devex comes later; this
    // is the version its iteration counts get compared against.
    //
    // Unless the solver is stalling, in which case the first eligible column by
    // index is taken instead. That is Bland's rule, and it is the half of the
    // anti-cycling guarantee that lives here; the other half is in the ratio
    // test below. Both ends have to follow it for the guarantee to hold.
    Int entering = -1;
    double best = 0.0;
    bool increase = true;
    for (Int j = 0; j < columns; ++j) {
      const VarStatus st = basis.status()[sz(j)];
      if (st == VarStatus::kBasic) continue;
      const double d = reduced[sz(j)];
      if (bland) {
        if (st == VarStatus::kAtLower && d < -options.dual_tolerance) {
          entering = j; increase = true; break;
        }
        if (st == VarStatus::kAtUpper && d > options.dual_tolerance) {
          entering = j; increase = false; break;
        }
        if (st == VarStatus::kFree && std::fabs(d) > options.dual_tolerance) {
          entering = j; increase = d < 0.0; break;
        }
        continue;
      }
      // Eligibility is on the reduced cost itself; only the ranking is
      // normalised. A column that does not improve the objective must never be
      // made attractive by a small weight.
      double magnitude = 0.0;
      bool up = true;
      if (st == VarStatus::kAtLower && d < -options.dual_tolerance) {
        magnitude = -d;
        up = true;
      } else if (st == VarStatus::kAtUpper && d > options.dual_tolerance) {
        magnitude = d;
        up = false;
      } else if (st == VarStatus::kFree && std::fabs(d) > options.dual_tolerance) {
        magnitude = std::fabs(d);
        up = d < 0.0;
      } else {
        continue;
      }

      const double score = options.pricing == SimplexOptions::Pricing::kDevex
                               ? magnitude * magnitude / std::fmax(weight[sz(j)], 1e-12)
                               : magnitude;
      if (score > best) {
        best = score;
        entering = j;
        increase = up;
      }
    }

    if (entering < 0) {
      // Both conclusions available here - that the model is infeasible, and
      // that this point is optimal - are claims about every column, drawn from
      // reduced costs computed through the product form. An updated basis is
      // accurate enough to pivot on and not always accurate enough to close a
      // case with: on brandy a stale one priced every column as unimprovable
      // and the solver reported a feasible model infeasible. So nothing is
      // concluded until the factorisation is fresh. If it already is, the
      // conclusion stands.
      if (basis.updates_since_refactorization() > 0) {
        if (basis.refactorize(form, &error)) {
          result.refactorizations++;
          ++result.confirmations;
          continue;
        }
        status = SimplexStatus::kNumericalError;
        result.message = "could not refactorise to confirm the result: " + error;
        break;
      }
      if (phase_one) {
        // Nothing improves the infeasibility, so there is no feasible point.
        status = SimplexStatus::kInfeasible;
      } else {
        status = SimplexStatus::kOptimal;
      }
      break;
    }

    basis.ftran_column(form, entering, &column);
    const double direction = increase ? 1.0 : -1.0;

    // Ratio test. Moving the entering variable by t changes basic i by
    // -direction * column[i] * t, and the step is limited by whichever basic
    // reaches a bound first - or by the entering variable's own opposite bound.
    //
    // This takes the smallest ratio and accepts whatever pivot element comes
    // with it. Harris's two-pass test was tried here instead: relax every bound
    // by the feasibility tolerance to find a slightly larger limit, then among
    // every row within that limit take the largest pivot. It is the standard
    // answer to exactly the problem brandy has, and measured over fourteen
    // Netlib instances it was a net loss - three better, six worse, and blend
    // stopped solving at all. The reason is in the literature and was skipped
    // on the way in: Harris lets basic variables overshoot their bounds by a
    // little, and needs Gill, Murray, Saunders and Wright's EXPAND (Math. Prog.
    // 45, 1989) alongside it to stop those overshoots accumulating. Half of a
    // two-part method is not a smaller version of it. Worth doing properly, not
    // worth doing like this.
    double step = form.upper[sz(entering)] - form.lower[sz(entering)];
    if (!std::isfinite(step)) step = kInf;
    Int leaving_row = -1;
    VarStatus leaving_to = VarStatus::kAtLower;

    for (Int i = 0; i < m; ++i) {
      const double alpha = direction * column[sz(i)];
      if (std::fabs(alpha) < options.pivot_tolerance) continue;
      const Int j = basis.basic()[sz(i)];
      const double value = z[sz(j)];
      const double lo = form.lower[sz(j)];
      const double hi = form.upper[sz(j)];

      // A basic variable that is already outside its bounds is heading back in;
      // it blocks when it arrives, not when it leaves the far side.
      double limit = kInf;
      VarStatus to = VarStatus::kAtLower;
      if (alpha > 0.0) {
        if (value < lo - options.primal_tolerance) {
          limit = (value - lo) / -alpha;  // negative distance, moving up
          to = VarStatus::kAtLower;
        } else if (std::isfinite(lo)) {
          limit = (value - lo) / alpha;
          to = VarStatus::kAtLower;
        }
      } else {
        if (value > hi + options.primal_tolerance) {
          limit = (value - hi) / -alpha;
          to = VarStatus::kAtUpper;
        } else if (std::isfinite(hi)) {
          limit = (hi - value) / -alpha;
          to = VarStatus::kAtUpper;
        }
      }
      if (limit < 0.0) limit = 0.0;
      if (limit < step - kRatioTie) {
        step = limit;
        leaving_row = i;
        leaving_to = to;
      } else if (bland && leaving_row >= 0 && limit <= step + kRatioTie &&
                 basis.basic()[sz(i)] < basis.basic()[sz(leaving_row)]) {
        // Ties in the ratio go to the lowest variable index, not the lowest
        // row. Breaking them by row is what lets a cycle close.
        step = std::fmin(step, limit);
        leaving_row = i;
        leaving_to = to;
      }
    }

    if (leaving_row < 0) {
      if (!std::isfinite(step)) {
        status = phase_one ? SimplexStatus::kInfeasible : SimplexStatus::kUnbounded;
        break;
      }
      // A bound flip: the entering variable moves to its other bound and the
      // basis is unchanged.
      basis.status()[sz(entering)] = increase ? VarStatus::kAtUpper
                                              : VarStatus::kAtLower;
      continue;
    }

    // Devex update, before the basis changes, because the pivot row it needs is
    // a row of the current basis inverse.
    //
    //   w_j  <- max(w_j, (alpha_rj / alpha_r)^2 * w_q)   for nonbasic j
    //   w_r  <- max(w_q / alpha_r^2, 1)                  for the one leaving
    //
    // alpha_rj is the pivot row of the tableau, rho' a_j, and alpha_r is the
    // pivot itself. The weights only ever rise within a framework, which is
    // what keeps the estimate an upper bound on the true edge norm and what
    // makes the reset below necessary.
    if (options.pricing == SimplexOptions::Pricing::kDevex) {
      const double alpha_r = column[sz(leaving_row)];
      if (std::fabs(alpha_r) > 1e-12) {
        basis.pivot_row(form, leaving_row, &rho);
        const double wq = std::fmax(weight[sz(entering)], 1.0);
        const double inv = 1.0 / alpha_r;
        double largest = 0.0;
        for (Int j = 0; j < columns; ++j) {
          if (j == entering) continue;
          if (basis.status()[sz(j)] == VarStatus::kBasic) continue;
          double alpha_rj = 0.0;
          for (Int e = form.columns.row_begin(j); e < form.columns.row_end(j); ++e) {
            alpha_rj += form.columns.value()[sz(e)] *
                        rho[sz(form.columns.index()[sz(e)])];
          }
          if (alpha_rj == 0.0) continue;
          const double ratio = alpha_rj * inv;
          const double candidate = ratio * ratio * wq;
          if (candidate > weight[sz(j)]) weight[sz(j)] = candidate;
          largest = std::fmax(largest, weight[sz(j)]);
        }
        const double leaving_weight = std::fmax(wq * inv * inv, 1.0);
        weight[sz(basis.basic()[sz(leaving_row)])] = leaving_weight;
        weight[sz(entering)] = 1.0;
        largest = std::fmax(largest, leaving_weight);

        // Past the threshold the estimates have drifted too far from the true
        // norms to rank anything usefully, so the framework restarts here.
        if (largest > options.devex_reset_weight) {
          std::fill(weight.begin(), weight.end(), 1.0);
          ++result.devex_resets;
        }
      }
    }

    if (step > options.degenerate_step) {
      degenerate_run = 0;
      if (bland) {
        bland = false;
        ++result.bland_switches;
      }
    } else if (++degenerate_run >= options.stall_iterations && !bland) {
      bland = true;
      ++result.bland_switches;
    }

    if (!basis.pivot(form, leaving_row, entering, leaving_to, column, &error)) {
      if (!basis.refactorize(form, &error)) {
        status = SimplexStatus::kNumericalError;
        result.message = "basis became singular: " + error;
        break;
      }
      result.refactorizations++;
      continue;
    }

    if (options.verbose && (iteration + 1) % options.log_frequency == 0) {
      std::printf("  iter %7d  %s  entering %6d  leaving row %6d  step %.3e\n",
                  iteration + 1, phase_one ? "phase 1" : "phase 2", entering,
                  leaving_row, step);
    }
  }

  basis.compute_primal(form, &z);
  basis.compute_duals(form, &duals, &reduced);

  result.x.assign(sz(form.num_structural), 0.0);
  for (Int j = 0; j < form.num_structural; ++j) result.x[sz(j)] = z[sz(j)];
  result.y = duals;

  double standard = 0.0;
  for (Int j = 0; j < form.num_structural; ++j) standard += form.cost[sz(j)] * z[sz(j)];
  result.objective = form.objective_scale * standard + form.objective_offset;
  result.iterations = iteration;
  result.status = status;
  result.solve_seconds = elapsed();
  return result;
}

}  // namespace sankhya
