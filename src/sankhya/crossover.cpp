#include "sankhya/crossover.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace sankhya {
namespace {

// How far inside its own bounds the point has this column, relative to the range
// it has to move in. A column at a bound scores zero and is not a candidate; a
// free column scores large, because a free column has no bound to sit at and is
// basic in every basis that is not degenerate.
double interior_score(double value, double lower, double upper) {
  const bool has_lower = lower > -kInf;
  const bool has_upper = upper < kInf;
  if (!has_lower && !has_upper) return kInf;

  double slack = kInf;
  if (has_lower) slack = std::fmin(slack, value - lower);
  if (has_upper) slack = std::fmin(slack, upper - value);
  if (slack <= 0.0) return 0.0;

  double scale = 1.0;
  if (has_lower && has_upper) {
    scale = std::fmax(1.0, upper - lower);
  } else {
    scale = std::fmax(1.0, std::fabs(value));
  }
  return slack / scale;
}

}  // namespace

CrossoverResult crossover_basis(const StandardLp& lp, const std::vector<double>& x,
                                const std::vector<double>& y,
                                const CrossoverOptions& options) {
  CrossoverResult result;
  const Int n = lp.num_cols();
  const Int m = lp.num_rows();
  if (static_cast<Int>(x.size()) != n) {
    result.message = "the point does not match the problem";
    return result;
  }

  const LogicalForm form = to_logical_form(lp);
  SimplexBasis basis;
  std::string error;
  if (!basis.set_initial(form, &error)) {
    result.message = error;
    return result;
  }

  // The logical of row i takes the row's slack, since a'x - s = q. A row the
  // point has tight has s near zero, so its logical is sitting at its own bound
  // and is exactly the one worth displacing; a row with real slack wants to
  // keep its logical basic.
  std::vector<double> kx(sz(m), 0.0);
  lp.k.multiply(x.data(), kx.data());
  std::vector<double> row_slack(sz(m), 0.0);
  for (Int i = 0; i < m; ++i) {
    row_slack[sz(i)] = std::fabs(kx[sz(i)] - lp.q[sz(i)]);
  }

  // Structural columns the point has strictly inside their bounds, best first.
  std::vector<std::pair<double, Int>> candidates;
  for (Int j = 0; j < n; ++j) {
    const double score = interior_score(x[sz(j)], lp.lower[sz(j)], lp.upper[sz(j)]);
    if (score > options.interior_tolerance) candidates.push_back({score, j});
  }
  std::sort(candidates.begin(), candidates.end(),
            [](const std::pair<double, Int>& a, const std::pair<double, Int>& b) {
              if (a.first != b.first) return a.first > b.first;
              return a.second < b.second;
            });
  if (options.max_candidates >= 0 &&
      static_cast<Int>(candidates.size()) > options.max_candidates) {
    candidates.resize(sz(options.max_candidates));
  }
  result.candidates = static_cast<Int>(candidates.size());

  // Push them in one at a time. Each pivot preserves nonsingularity in exact
  // arithmetic; in floating point the product form still decays, so the last
  // basis that actually factorised from scratch is kept and is what gets handed
  // over if a later one will not.
  std::vector<double> alpha;
  std::vector<Int> good_basic = basis.basic();
  std::vector<VarStatus> good_status = basis.status();
  auto restore_last_good = [&]() {
    std::string ignored;
    basis.set_from(form, good_basic, good_status, &ignored);
  };
  Int logicals_basic = m;
  for (const auto& candidate : candidates) {
    if (logicals_basic == 0) break;
    const Int column = candidate.second;
    basis.ftran_column(form, column, &alpha);

    // Among rows still held by a logical, take the largest pivot, preferring
    // rows the point has tight - those are the logicals that should be leaving
    // anyway, and displacing one costs the point nothing.
    Int best_row = -1;
    double best_merit = 0.0;
    for (Int i = 0; i < m; ++i) {
      const Int occupant = basis.basic()[sz(i)];
      if (occupant < form.num_structural) continue;  // already a structural
      const double pivot = std::fabs(alpha[sz(i)]);
      if (!(pivot > options.pivot_tolerance)) continue;
      const double merit = pivot / (1.0 + row_slack[sz(i)]);
      if (merit > best_merit) {
        best_merit = merit;
        best_row = i;
      }
    }
    if (best_row < 0) {
      result.rejected_small_pivot++;
      continue;
    }
    if (!basis.pivot(form, best_row, column, VarStatus::kAtLower, alpha, &error)) {
      result.rejected_small_pivot++;
      continue;
    }
    result.pushed++;
    logicals_basic--;

    // Redo the factorisation periodically. A run of pushes that only works
    // through the accumulated updates is a basis the simplex will reject on
    // arrival, because installing one factorises it from scratch.
    if (options.refactor_interval > 0 &&
        basis.updates_since_refactorization() >= options.refactor_interval) {
      if (basis.refactorize(form, &error)) {
        good_basic = basis.basic();
        good_status = basis.status();
      } else {
        // Go back to the last basis that did factorise and stop there. A
        // shorter good basis beats a longer decayed one, and this is the only
        // exit that cannot fail - that basis factorised once already.
        restore_last_good();
        result.stopped_on_refactorization = true;
        break;
      }
    }
  }

  // One last factorisation before handing it over, for the same reason: the
  // simplex will factorise this basis from scratch when it installs it, so a
  // basis that only works through the leftover updates gets rejected on arrival
  // and the whole exercise buys nothing.
  //
  // cycle is the instance that made this necessary. 749 columns pushed, every
  // interior check passed, and the simplex refused the result and started cold
  // - so the crossover reported success while quietly doing nothing at all.
  if (!basis.refactorize(form, &error)) {
    restore_last_good();
    result.stopped_on_refactorization = true;
  }

  // Recount from the basis that is actually being handed over, which after a
  // restore is not the one the loop finished on.
  result.pushed = 0;
  logicals_basic = 0;
  for (Int i = 0; i < m; ++i) {
    if (basis.basic()[sz(i)] < form.num_structural) {
      result.pushed++;
    } else {
      logicals_basic++;
    }
  }
  result.logicals_remaining = logicals_basic;

  // Everything not basic has to sit somewhere, and the point says where. A
  // column the point has at or below its lower bound goes to the lower bound;
  // one at or above its upper goes to the upper. In between - which happens for
  // a candidate whose pivot was refused - the reduced cost decides if we have a
  // dual to compute it from, and proximity decides if we do not.
  std::vector<double> reduced_cost;
  if (static_cast<Int>(y.size()) == m) {
    reduced_cost.assign(sz(n), 0.0);
    lp.kt.multiply(y.data(), reduced_cost.data());
    for (Int j = 0; j < n; ++j) reduced_cost[sz(j)] = lp.c[sz(j)] - reduced_cost[sz(j)];
  }

  std::vector<VarStatus>& status = basis.status();
  for (Int j = 0; j < n; ++j) {
    if (status[sz(j)] == VarStatus::kBasic) continue;
    const double lower = lp.lower[sz(j)];
    const double upper = lp.upper[sz(j)];
    if (lower <= -kInf && upper >= kInf) {
      status[sz(j)] = VarStatus::kFree;
      continue;
    }
    if (lower <= -kInf) {
      status[sz(j)] = VarStatus::kAtUpper;
      continue;
    }
    if (upper >= kInf) {
      status[sz(j)] = VarStatus::kAtLower;
      continue;
    }
    bool prefer_upper = (x[sz(j)] - lower) > (upper - x[sz(j)]);
    if (!reduced_cost.empty()) {
      // Minimisation: a negative reduced cost wants the variable up.
      if (reduced_cost[sz(j)] < 0.0) prefer_upper = true;
      if (reduced_cost[sz(j)] > 0.0) prefer_upper = false;
    }
    status[sz(j)] = prefer_upper ? VarStatus::kAtUpper : VarStatus::kAtLower;
  }
  // A logical that did not stay basic sits at zero, which is its lower bound
  // whether the row is an equality or an inequality.
  for (Int i = 0; i < m; ++i) {
    const Int column = form.num_structural + i;
    if (status[sz(column)] != VarStatus::kBasic) status[sz(column)] = VarStatus::kAtLower;
  }

  result.basic = basis.basic();
  result.status = status;
  result.ok = true;
  return result;
}

}  // namespace sankhya
