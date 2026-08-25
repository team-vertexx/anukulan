#include "sankhya/presolve.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <unordered_map>

namespace sankhya {

std::string to_string(PresolveStatus status) {
  switch (status) {
    case PresolveStatus::kReduced: return "reduced";
    case PresolveStatus::kInfeasible: return "infeasible";
    case PresolveStatus::kUnbounded: return "unbounded";
  }
  return "unknown";
}

void PostsolveStack::record_fixed(Int col, double value) {
  Entry e;
  e.kind = Kind::kFixed;
  e.col = col;
  e.value = value;
  entries_.push_back(std::move(e));
}

void PostsolveStack::record_singleton(Int col, double coefficient, double rhs,
                                      std::vector<Term> other_terms) {
  Entry e;
  e.kind = Kind::kSingleton;
  e.col = col;
  e.value = rhs;
  e.coefficient = coefficient;
  e.terms = std::move(other_terms);
  entries_.push_back(std::move(e));
}

void PostsolveStack::set_dimensions(Int original_cols,
                                    std::vector<Int> reduced_to_original) {
  original_cols_ = original_cols;
  reduced_to_original_ = std::move(reduced_to_original);
}

std::vector<double> PostsolveStack::apply(const std::vector<double>& reduced_x) const {
  std::vector<double> x(sz(original_cols_), 0.0);
  const std::size_t n = std::min(reduced_x.size(), reduced_to_original_.size());
  for (std::size_t j = 0; j < n; ++j) x[sz(reduced_to_original_[j])] = reduced_x[j];

  // Reverse order matters. A column eliminated from an equality row is read off
  // the other columns of that row, and any of those that were themselves removed
  // were removed later, so replaying backwards has already restored them.
  for (std::size_t e = entries_.size(); e-- > 0;) {
    const Entry& entry = entries_[e];
    if (entry.kind == Kind::kFixed) {
      x[sz(entry.col)] = entry.value;
    } else {
      double rest = 0.0;
      for (const Term& t : entry.terms) rest += t.coefficient * x[sz(t.col)];
      x[sz(entry.col)] = (entry.value - rest) / entry.coefficient;
    }
  }
  return x;
}

namespace {

// Everything the reductions read and write. A's coefficients are never touched,
// so the original CSR and its transpose stay valid for the whole run and a
// removal is a flag plus a decremented count.
struct Workspace {
  const SparseMatrix* a = nullptr;   // rows x cols, CSR
  SparseMatrix at;                   // cols x rows, the column view

  std::vector<char> row_alive;
  std::vector<char> col_alive;
  std::vector<Int> row_count;  // live entries in the row
  std::vector<Int> col_count;  // live entries in the column

  std::vector<double> rlo, rup;
  std::vector<double> clo, cup;
  std::vector<double> cost;  // in the minimise direction
  std::vector<char> integral;
  double cost_offset = 0.0;  // also in the minimise direction

  PostsolveStack* stack = nullptr;
  PresolveCounts* counts = nullptr;
  double tol = 1e-9;    // a reduction may fire
  double infeas = 1e-7;  // presolve may say "infeasible"
  double relax = 0.0;
  double max_new_finite = 1e7;

  bool infeasible = false;
  bool unbounded = false;
  std::string message;

  Int rows() const { return a->rows(); }
  Int cols() const { return a->cols(); }

  void fail(const std::string& why) {
    if (!infeasible && !unbounded) message = why;
    infeasible = true;
  }
};

double scale_of(double v) { return std::fmax(1.0, std::fabs(v)); }

void kill_row(Workspace& w, Int i) {
  if (!w.row_alive[sz(i)]) return;
  w.row_alive[sz(i)] = 0;
  for (Int e = w.a->row_begin(i); e < w.a->row_end(i); ++e) {
    const Int j = w.a->index()[sz(e)];
    if (w.col_alive[sz(j)]) {
      --w.col_count[sz(j)];
      ++w.counts->nonzeros_removed;
    }
  }
  w.row_count[sz(i)] = 0;
  ++w.counts->rows_removed;
}

void kill_col(Workspace& w, Int j) {
  if (!w.col_alive[sz(j)]) return;
  w.col_alive[sz(j)] = 0;
  for (Int e = w.at.row_begin(j); e < w.at.row_end(j); ++e) {
    const Int i = w.at.index()[sz(e)];
    if (w.row_alive[sz(i)]) {
      --w.row_count[sz(i)];
      ++w.counts->nonzeros_removed;
    }
  }
  w.col_count[sz(j)] = 0;
  ++w.counts->cols_removed;
}

// Substitute a known value out of every row it appears in. This is the only
// place a row bound moves for a reason other than tightening.
void fix_column(Workspace& w, Int j, double value) {
  for (Int e = w.at.row_begin(j); e < w.at.row_end(j); ++e) {
    const Int i = w.at.index()[sz(e)];
    if (!w.row_alive[sz(i)]) continue;
    const double a = w.at.value()[sz(e)];
    if (!std::isinf(w.rlo[sz(i)])) w.rlo[sz(i)] -= a * value;
    if (!std::isinf(w.rup[sz(i)])) w.rup[sz(i)] -= a * value;
  }
  w.cost_offset += w.cost[sz(j)] * value;
  w.stack->record_fixed(j, value);
  kill_col(w, j);
}

bool tighten_lower(Workspace& w, Int j, double value) {
  if (!std::isfinite(value)) return false;
  double v = value;
  if (w.integral[sz(j)]) {
    v = std::ceil(v - 1e-9);
  } else {
    v -= w.relax * scale_of(v);
  }
  const double current = w.clo[sz(j)];
  if (std::isinf(current) && std::fabs(v) > w.max_new_finite) return false;
  const double gate = std::isinf(current) ? 0.0 : 1e-7 * scale_of(current);
  if (!(v > current + gate)) return false;
  if (std::isinf(current)) {
    ++w.counts->bounds_made_finite;
    w.counts->largest_new_finite_bound =
        std::fmax(w.counts->largest_new_finite_bound, std::fabs(v));
  }
  w.clo[sz(j)] = v;
  ++w.counts->bounds_tightened;
  if (w.clo[sz(j)] > w.cup[sz(j)] + w.infeas * scale_of(w.cup[sz(j)]))
    w.fail("bound tightening crossed the bounds of a column");
  return true;
}

bool tighten_upper(Workspace& w, Int j, double value) {
  if (!std::isfinite(value)) return false;
  double v = value;
  if (w.integral[sz(j)]) {
    v = std::floor(v + 1e-9);
  } else {
    v += w.relax * scale_of(v);
  }
  const double current = w.cup[sz(j)];
  if (std::isinf(current) && std::fabs(v) > w.max_new_finite) return false;
  const double gate = std::isinf(current) ? 0.0 : 1e-7 * scale_of(current);
  if (!(v < current - gate)) return false;
  if (std::isinf(current)) {
    ++w.counts->bounds_made_finite;
    w.counts->largest_new_finite_bound =
        std::fmax(w.counts->largest_new_finite_bound, std::fabs(v));
  }
  w.cup[sz(j)] = v;
  ++w.counts->bounds_tightened;
  if (w.clo[sz(j)] > w.cup[sz(j)] + w.infeas * scale_of(w.cup[sz(j)]))
    w.fail("bound tightening crossed the bounds of a column");
  return true;
}

// Interval arithmetic over one row's live entries.
struct Activity {
  double min_value = 0.0;
  double max_value = 0.0;
  Int min_infinite = 0;
  Int max_infinite = 0;
  double max_abs_coeff = 0.0;
};

Activity row_activity(const Workspace& w, Int i) {
  Activity act;
  for (Int e = w.a->row_begin(i); e < w.a->row_end(i); ++e) {
    const Int j = w.a->index()[sz(e)];
    if (!w.col_alive[sz(j)]) continue;
    const double a = w.a->value()[sz(e)];
    act.max_abs_coeff = std::fmax(act.max_abs_coeff, std::fabs(a));
    const double lo = (a > 0.0) ? w.clo[sz(j)] : w.cup[sz(j)];
    const double hi = (a > 0.0) ? w.cup[sz(j)] : w.clo[sz(j)];
    if (std::isinf(lo)) ++act.min_infinite; else act.min_value += a * lo;
    if (std::isinf(hi)) ++act.max_infinite; else act.max_value += a * hi;
  }
  return act;
}

}  // namespace

namespace {

// One pass over the rows: empty, singleton, forcing, redundant.
bool scan_rows(Workspace& w, const PresolveOptions& opt) {
  bool changed = false;
  for (Int i = 0; i < w.rows(); ++i) {
    if (!w.row_alive[sz(i)] || w.infeasible) continue;

    const double lo = w.rlo[sz(i)];
    const double hi = w.rup[sz(i)];
    if (lo > hi + w.infeas * scale_of(hi)) {
      w.fail("a row's lower bound rose above its upper bound");
      return changed;
    }

    if (w.row_count[sz(i)] == 0) {
      if (!opt.empty_rows) continue;
      // Nothing left to satisfy it with, so the row says 0 in [lo, hi].
      if (lo > w.infeas * scale_of(lo) || hi < -w.infeas * scale_of(hi)) {
        w.fail("an empty row cannot hold zero");
        return changed;
      }
      ++w.counts->empty_rows;
      kill_row(w, i);
      changed = true;
      continue;
    }

    const Activity act = row_activity(w, i);
    const double lo_scale = std::isinf(lo) ? 1.0 : scale_of(lo);
    const double hi_scale = std::isinf(hi) ? 1.0 : scale_of(hi);

    if (act.min_infinite == 0 && !std::isinf(hi) &&
        act.min_value > hi + w.infeas * hi_scale) {
      w.fail("a row's smallest possible activity exceeds its upper bound");
      return changed;
    }
    if (act.max_infinite == 0 && !std::isinf(lo) &&
        act.max_value < lo - w.infeas * lo_scale) {
      w.fail("a row's largest possible activity is below its lower bound");
      return changed;
    }

    if (w.row_count[sz(i)] == 1 && opt.singleton_rows) {
      // The row is a bound wearing a row's clothes. Fold it into the column and
      // drop it; this is exact, not an approximation, so no relaxation is owed.
      Int col = -1;
      double a = 0.0;
      for (Int e = w.a->row_begin(i); e < w.a->row_end(i); ++e) {
        const Int j = w.a->index()[sz(e)];
        if (!w.col_alive[sz(j)]) continue;
        col = j;
        a = w.a->value()[sz(e)];
        break;
      }
      if (col >= 0 && std::fabs(a) > 1e-30) {
        const double b1 = std::isinf(lo) ? (a > 0.0 ? -kInf : kInf) : lo / a;
        const double b2 = std::isinf(hi) ? (a > 0.0 ? kInf : -kInf) : hi / a;
        tighten_lower(w, col, std::fmin(b1, b2));
        tighten_upper(w, col, std::fmax(b1, b2));
        ++w.counts->singleton_rows;
        kill_row(w, i);
        changed = true;
        continue;
      }
    }

    // Forcing: the row can only be met at one end of its activity range, so
    // every variable in it is pinned. Checked before redundancy because it is
    // strictly the stronger conclusion.
    if (opt.forcing_rows && act.min_infinite == 0 && !std::isinf(hi) &&
        act.min_value >= hi - w.tol * hi_scale) {
      for (Int e = w.a->row_begin(i); e < w.a->row_end(i); ++e) {
        const Int j = w.a->index()[sz(e)];
        if (!w.col_alive[sz(j)]) continue;
        const double a = w.a->value()[sz(e)];
        fix_column(w, j, (a > 0.0) ? w.clo[sz(j)] : w.cup[sz(j)]);
      }
      ++w.counts->forcing_rows;
      kill_row(w, i);
      changed = true;
      continue;
    }
    if (opt.forcing_rows && act.max_infinite == 0 && !std::isinf(lo) &&
        act.max_value <= lo + w.tol * lo_scale) {
      for (Int e = w.a->row_begin(i); e < w.a->row_end(i); ++e) {
        const Int j = w.a->index()[sz(e)];
        if (!w.col_alive[sz(j)]) continue;
        const double a = w.a->value()[sz(e)];
        fix_column(w, j, (a > 0.0) ? w.cup[sz(j)] : w.clo[sz(j)]);
      }
      ++w.counts->forcing_rows;
      kill_row(w, i);
      changed = true;
      continue;
    }

    if (opt.redundant_rows) {
      const bool lower_ok =
          std::isinf(lo) || (act.min_infinite == 0 && act.min_value >= lo - w.tol * lo_scale);
      const bool upper_ok =
          std::isinf(hi) || (act.max_infinite == 0 && act.max_value <= hi + w.tol * hi_scale);
      if (lower_ok && upper_ok) {
        ++w.counts->redundant_rows;
        kill_row(w, i);
        changed = true;
        continue;
      }
    }

    if (opt.bound_tightening) {
      for (Int e = w.a->row_begin(i); e < w.a->row_end(i); ++e) {
        const Int j = w.a->index()[sz(e)];
        if (!w.col_alive[sz(j)]) continue;
        const double a = w.a->value()[sz(e)];
        if (std::fabs(a) < 1e-12 * std::fmax(1.0, act.max_abs_coeff)) continue;

        const double own_min = (a > 0.0) ? w.clo[sz(j)] : w.cup[sz(j)];
        const double own_max = (a > 0.0) ? w.cup[sz(j)] : w.clo[sz(j)];

        // a x_j <= hi - (smallest the rest of the row can be)
        if (!std::isinf(hi) &&
            (act.min_infinite == 0 || (act.min_infinite == 1 && std::isinf(own_min)))) {
          const double rest =
              act.min_infinite == 0 ? act.min_value - a * own_min : act.min_value;
          const double limit = (hi - rest) / a;
          if (a > 0.0) { if (tighten_upper(w, j, limit)) changed = true; }
          else { if (tighten_lower(w, j, limit)) changed = true; }
        }
        // a x_j >= lo - (largest the rest of the row can be)
        if (!std::isinf(lo) &&
            (act.max_infinite == 0 || (act.max_infinite == 1 && std::isinf(own_max)))) {
          const double rest =
              act.max_infinite == 0 ? act.max_value - a * own_max : act.max_value;
          const double limit = (lo - rest) / a;
          if (a > 0.0) { if (tighten_lower(w, j, limit)) changed = true; }
          else { if (tighten_upper(w, j, limit)) changed = true; }
        }
        if (w.infeasible) return changed;
      }
    }
  }
  return changed;
}

// One pass over the columns: fixed, empty, and free singletons.
bool scan_columns(Workspace& w, const PresolveOptions& opt, bool allow_removal) {
  if (!allow_removal) return false;
  bool changed = false;
  for (Int j = 0; j < w.cols(); ++j) {
    if (!w.col_alive[sz(j)] || w.infeasible) continue;

    if (w.clo[sz(j)] > w.cup[sz(j)] + w.infeas * scale_of(w.cup[sz(j)])) {
      w.fail("a column's bounds crossed");
      return changed;
    }

    if (opt.fixed_columns &&
        w.cup[sz(j)] - w.clo[sz(j)] <= w.tol * scale_of(w.clo[sz(j)]) &&
        !std::isinf(w.clo[sz(j)])) {
      ++w.counts->fixed_columns;
      fix_column(w, j, w.clo[sz(j)]);
      changed = true;
      continue;
    }

    if (w.col_count[sz(j)] == 0) {
      if (!opt.empty_columns) continue;
      // In no row at all, so only the objective has an opinion.
      const double c = w.cost[sz(j)];
      double value = 0.0;
      if (c > 0.0) {
        if (std::isinf(w.clo[sz(j)])) {
          w.unbounded = true;
          w.message = "a column in no row can decrease the objective forever";
          return changed;
        }
        value = w.clo[sz(j)];
      } else if (c < 0.0) {
        if (std::isinf(w.cup[sz(j)])) {
          w.unbounded = true;
          w.message = "a column in no row can decrease the objective forever";
          return changed;
        }
        value = w.cup[sz(j)];
      } else {
        value = std::fmin(std::fmax(0.0, w.clo[sz(j)]), w.cup[sz(j)]);
        if (std::isinf(value)) value = 0.0;
      }
      ++w.counts->empty_columns;
      fix_column(w, j, value);
      changed = true;
      continue;
    }

    if (w.col_count[sz(j)] == 1 && opt.free_column_singletons && !w.integral[sz(j)]) {
      Int row = -1;
      double a = 0.0;
      for (Int e = w.at.row_begin(j); e < w.at.row_end(j); ++e) {
        const Int i = w.at.index()[sz(e)];
        if (!w.row_alive[sz(i)]) continue;
        row = i;
        a = w.at.value()[sz(e)];
        break;
      }
      if (row < 0) continue;
      const double lo = w.rlo[sz(row)];
      const double hi = w.rup[sz(row)];
      if (std::isinf(lo) || std::isinf(hi)) continue;
      if (std::fabs(hi - lo) > w.tol * scale_of(hi)) continue;  // not an equality

      const Activity act = row_activity(w, row);
      if (std::fabs(a) < 1e-7 * act.max_abs_coeff || std::fabs(a) < 1e-10) continue;

      // The column is substitutable when its own bounds cannot bind: whatever the
      // rest of the row does, the value the equality forces on x_j already lies
      // inside them. A genuinely free column passes this trivially.
      const bool free_below = std::isinf(w.clo[sz(j)]);
      const bool free_above = std::isinf(w.cup[sz(j)]);
      if (!(free_below && free_above)) {
        // Both activity ends have to be finite for the implied interval to
        // exist. That also makes this column's own contribution finite, since
        // it is one of the terms being summed.
        if (act.min_infinite > 0 || act.max_infinite > 0) continue;
        const double own_min = (a > 0.0) ? w.clo[sz(j)] : w.cup[sz(j)];
        const double own_max = (a > 0.0) ? w.cup[sz(j)] : w.clo[sz(j)];
        const double rest_min = act.min_value - a * own_min;
        const double rest_max = act.max_value - a * own_max;
        const double e1 = (lo - rest_max) / a;
        const double e2 = (lo - rest_min) / a;
        const double implied_lo = std::fmin(e1, e2);
        const double implied_hi = std::fmax(e1, e2);
        const double slack = w.tol * scale_of(implied_hi);
        if (!(implied_lo >= w.clo[sz(j)] - slack && implied_hi <= w.cup[sz(j)] + slack))
          continue;
      }

      std::vector<PostsolveStack::Term> terms;
      for (Int e = w.a->row_begin(row); e < w.a->row_end(row); ++e) {
        const Int k = w.a->index()[sz(e)];
        if (k == j || !w.col_alive[sz(k)]) continue;
        terms.push_back({k, w.a->value()[sz(e)]});
      }
      // x_j = (b - sum_k a_k x_k) / a, so the objective loses c_j x_j and gains
      // a constant plus a correction on every other column of the row.
      const double cj = w.cost[sz(j)];
      if (cj != 0.0) {
        w.cost_offset += cj * lo / a;
        for (const PostsolveStack::Term& t : terms)
          w.cost[sz(t.col)] -= cj * t.coefficient / a;
      }
      w.stack->record_singleton(j, a, lo, std::move(terms));
      ++w.counts->free_column_singletons;
      kill_col(w, j);
      kill_row(w, row);
      changed = true;
      continue;
    }
  }
  return changed;
}

double divide_bound(double bound, double r) {
  if (std::isinf(bound)) return (r > 0.0) ? bound : -bound;
  return bound / r;
}

// Rows that are scalar multiples of one another say the same thing twice. Hash
// the live column pattern, then confirm proportionality inside each bucket -
// hashing the values themselves would be at the mercy of the last bit.
bool merge_duplicate_rows(Workspace& w) {
  std::unordered_map<std::size_t, std::vector<Int>> buckets;
  std::vector<Int> pattern;
  for (Int i = 0; i < w.rows(); ++i) {
    if (!w.row_alive[sz(i)] || w.row_count[sz(i)] < 2) continue;
    pattern.clear();
    for (Int e = w.a->row_begin(i); e < w.a->row_end(i); ++e) {
      const Int j = w.a->index()[sz(e)];
      if (w.col_alive[sz(j)]) pattern.push_back(j);
    }
    std::size_t h = 1469598103934665603ull;
    for (Int j : pattern) {
      h ^= static_cast<std::size_t>(static_cast<std::uint32_t>(j));
      h *= 1099511628211ull;
    }
    buckets[h].push_back(i);
  }

  bool changed = false;
  std::vector<Int> cols_a, cols_b;
  std::vector<double> vals_a, vals_b;
  for (auto& [hash, rows] : buckets) {
    (void)hash;
    if (rows.size() < 2) continue;
    // Buckets are tiny in practice; the cap is a guard against a pathological
    // model turning this quadratic, not something normal files reach.
    const std::size_t cap = 64;
    const std::size_t limit = std::min(rows.size(), cap);
    for (std::size_t x = 0; x < limit; ++x) {
      const Int i1 = rows[x];
      if (!w.row_alive[sz(i1)]) continue;
      cols_a.clear(); vals_a.clear();
      for (Int e = w.a->row_begin(i1); e < w.a->row_end(i1); ++e) {
        const Int j = w.a->index()[sz(e)];
        if (!w.col_alive[sz(j)]) continue;
        cols_a.push_back(j);
        vals_a.push_back(w.a->value()[sz(e)]);
      }
      if (cols_a.empty()) continue;

      for (std::size_t y = x + 1; y < limit; ++y) {
        const Int i2 = rows[y];
        if (!w.row_alive[sz(i2)]) continue;
        cols_b.clear(); vals_b.clear();
        for (Int e = w.a->row_begin(i2); e < w.a->row_end(i2); ++e) {
          const Int j = w.a->index()[sz(e)];
          if (!w.col_alive[sz(j)]) continue;
          cols_b.push_back(j);
          vals_b.push_back(w.a->value()[sz(e)]);
        }
        if (cols_b.size() != cols_a.size() || cols_b != cols_a) continue;
        if (std::fabs(vals_a[0]) < 1e-30) continue;

        const double r = vals_b[0] / vals_a[0];
        if (std::fabs(r) < 1e-12 || !std::isfinite(r)) continue;
        bool proportional = true;
        for (std::size_t k = 1; k < vals_a.size(); ++k) {
          const double want = r * vals_a[k];
          if (std::fabs(vals_b[k] - want) >
              1e-9 * std::fmax(1.0, std::fabs(want))) {
            proportional = false;
            break;
          }
        }
        if (!proportional) continue;

        // row2 = r * row1, so row2's bounds are bounds on row1's activity too.
        const double b1 = divide_bound(w.rlo[sz(i2)], r);
        const double b2 = divide_bound(w.rup[sz(i2)], r);
        const double lo = std::fmin(b1, b2);
        const double hi = std::fmax(b1, b2);
        w.rlo[sz(i1)] = std::fmax(w.rlo[sz(i1)], lo);
        w.rup[sz(i1)] = std::fmin(w.rup[sz(i1)], hi);
        if (w.rlo[sz(i1)] > w.rup[sz(i1)] + w.infeas * scale_of(w.rup[sz(i1)])) {
          w.fail("two parallel rows contradict each other");
          return changed;
        }
        ++w.counts->duplicate_rows;
        kill_row(w, i2);
        changed = true;
      }
    }
  }
  return changed;
}

Model build_reduced(const Workspace& w, const Model& model, ObjSense sense,
                    std::vector<Int>* reduced_to_original) {
  std::vector<Int> new_row(sz(w.rows()), -1);
  std::vector<Int> new_col(sz(w.cols()), -1);
  Int rows = 0, cols = 0;
  for (Int i = 0; i < w.rows(); ++i)
    if (w.row_alive[sz(i)]) new_row[sz(i)] = rows++;
  reduced_to_original->clear();
  for (Int j = 0; j < w.cols(); ++j) {
    if (!w.col_alive[sz(j)]) continue;
    new_col[sz(j)] = cols++;
    reduced_to_original->push_back(j);
  }

  Model out;
  out.name = model.name;
  out.sense = sense;
  const double s = (sense == ObjSense::kMaximize) ? -1.0 : 1.0;
  out.objective_offset = model.objective_offset + s * w.cost_offset;

  std::vector<Triplet> entries;
  entries.reserve(sz(w.a->nnz()));
  for (Int i = 0; i < w.rows(); ++i) {
    if (!w.row_alive[sz(i)]) continue;
    for (Int e = w.a->row_begin(i); e < w.a->row_end(i); ++e) {
      const Int j = w.a->index()[sz(e)];
      if (!w.col_alive[sz(j)]) continue;
      entries.push_back({new_row[sz(i)], new_col[sz(j)], w.a->value()[sz(e)]});
    }
  }
  out.constraints = SparseMatrix::from_triplets(rows, cols, std::move(entries));

  out.objective.reserve(sz(cols));
  out.col_lower.reserve(sz(cols));
  out.col_upper.reserve(sz(cols));
  out.col_type.reserve(sz(cols));
  for (Int j = 0; j < w.cols(); ++j) {
    if (!w.col_alive[sz(j)]) continue;
    out.objective.push_back(s * w.cost[sz(j)]);
    out.col_lower.push_back(w.clo[sz(j)]);
    out.col_upper.push_back(w.cup[sz(j)]);
    out.col_type.push_back(model.col_type[sz(j)]);
    if (!model.col_names.empty()) out.col_names.push_back(model.col_names[sz(j)]);
  }
  out.row_lower.reserve(sz(rows));
  out.row_upper.reserve(sz(rows));
  for (Int i = 0; i < w.rows(); ++i) {
    if (!w.row_alive[sz(i)]) continue;
    out.row_lower.push_back(w.rlo[sz(i)]);
    out.row_upper.push_back(w.rup[sz(i)]);
    if (!model.row_names.empty()) out.row_names.push_back(model.row_names[sz(i)]);
  }

  // Column-removing reductions are switched off when a Hessian is present, so
  // by the time this runs every column is still alive and Q carries over as it
  // stands. The check is here so a future reduction that forgets that rule
  // fails loudly instead of returning a quietly wrong quadratic.
  if (model.has_hessian() && cols == model.num_cols()) out.hessian = model.hessian;
  return out;
}

}  // namespace

PresolveResult presolve(const Model& model, const PresolveOptions& options) {
  const auto started = std::chrono::steady_clock::now();
  PresolveResult result;
  result.original_rows = model.num_rows();
  result.original_cols = model.num_cols();
  result.original_nnz = model.constraints.nnz();

  std::string error;
  if (!model.validate(&error)) {
    result.status = PresolveStatus::kInfeasible;
    result.message = "model did not validate: " + error;
    return result;
  }

  Workspace w;
  w.a = &model.constraints;
  w.at = model.constraints.transpose();
  w.row_alive.assign(sz(model.num_rows()), 1);
  w.col_alive.assign(sz(model.num_cols()), 1);
  w.row_count.assign(sz(model.num_rows()), 0);
  w.col_count.assign(sz(model.num_cols()), 0);
  for (Int i = 0; i < model.num_rows(); ++i)
    w.row_count[sz(i)] = model.constraints.row_end(i) - model.constraints.row_begin(i);
  for (Int j = 0; j < model.num_cols(); ++j)
    w.col_count[sz(j)] = w.at.row_end(j) - w.at.row_begin(j);
  w.rlo = model.row_lower;
  w.rup = model.row_upper;
  w.clo = model.col_lower;
  w.cup = model.col_upper;
  w.integral.assign(sz(model.num_cols()), 0);
  for (Int j = 0; j < model.num_cols(); ++j)
    w.integral[sz(j)] = (model.col_type[sz(j)] == VarType::kInteger ||
                         model.col_type[sz(j)] == VarType::kSemiInteger)
                            ? 1 : 0;

  // An integer column written with fractional bounds - and files do this, often
  // as 0 <= x <= 2.5 meaning "at most two" - is rounded once at the start. The
  // reduction is exact and costs one pass, and without it branch and bound
  // spends a node rediscovering it.
  for (Int j = 0; j < model.num_cols(); ++j) {
    if (!w.integral[sz(j)]) continue;
    if (std::isfinite(w.clo[sz(j)])) {
      const double v = std::ceil(w.clo[sz(j)] - 1e-9);
      if (v > w.clo[sz(j)]) { w.clo[sz(j)] = v; ++result.counts.bounds_tightened; }
    }
    if (std::isfinite(w.cup[sz(j)])) {
      const double v = std::floor(w.cup[sz(j)] + 1e-9);
      if (v < w.cup[sz(j)]) { w.cup[sz(j)] = v; ++result.counts.bounds_tightened; }
    }
    if (w.clo[sz(j)] > w.cup[sz(j)] + options.infeasibility_tolerance) {
      result.status = PresolveStatus::kInfeasible;
      result.message = "an integer column has no whole number between its bounds";
      return result;
    }
  }

  const double s = (model.sense == ObjSense::kMaximize) ? -1.0 : 1.0;
  w.cost.resize(sz(model.num_cols()));
  for (Int j = 0; j < model.num_cols(); ++j) w.cost[sz(j)] = s * model.objective[sz(j)];

  w.stack = &result.postsolve;
  w.counts = &result.counts;
  w.tol = options.feasibility_tolerance;
  w.infeas = std::fmax(options.infeasibility_tolerance, options.feasibility_tolerance);
  w.relax = options.bound_relaxation;
  w.max_new_finite = options.max_new_finite_bound;

  const bool allow_removal = !model.has_hessian();

  for (Int round = 0; round < options.max_rounds; ++round) {
    ++result.counts.rounds;
    bool changed = false;
    changed |= scan_rows(w, options);
    if (w.infeasible || w.unbounded) break;
    changed |= scan_columns(w, options, allow_removal);
    if (w.infeasible || w.unbounded) break;
    if (options.duplicate_rows && round < 2) changed |= merge_duplicate_rows(w);
    if (w.infeasible || w.unbounded) break;
    if (!changed) break;
  }

  if (w.unbounded) {
    result.status = PresolveStatus::kUnbounded;
    result.message = w.message;
  } else if (w.infeasible) {
    result.status = PresolveStatus::kInfeasible;
    result.message = w.message;
  } else {
    result.status = PresolveStatus::kReduced;
    std::vector<Int> map;
    result.reduced = build_reduced(w, model, model.sense, &map);
    result.postsolve.set_dimensions(model.num_cols(), std::move(map));
  }

  const auto finished = std::chrono::steady_clock::now();
  result.seconds = std::chrono::duration<double>(finished - started).count();
  return result;
}

std::string format_presolve(const PresolveResult& result) {
  char buf[1400];
  const PresolveCounts& c = result.counts;
  const Int rows = result.original_rows - c.rows_removed;
  const Int cols = result.original_cols - c.cols_removed;
  const Int nnz = result.original_nnz - c.nonzeros_removed;
  std::snprintf(
      buf, sizeof(buf),
      "presolve      %s\n"
      "rows          %d -> %d\n"
      "columns       %d -> %d\n"
      "nonzeros      %d -> %d\n"
      "rounds        %d\n"
      "time          %.4f s\n"
      "removed by    empty row %d, singleton row %d, redundant row %d,\n"
      "              forcing row %d, duplicate row %d,\n"
      "              fixed column %d, empty column %d, free singleton %d\n"
      "bounds cut    %d  (%d were infinite before, largest now %.3e)\n",
      to_string(result.status).c_str(), result.original_rows, rows,
      result.original_cols, cols, result.original_nnz, nnz, c.rounds,
      result.seconds, c.empty_rows, c.singleton_rows, c.redundant_rows,
      c.forcing_rows, c.duplicate_rows, c.fixed_columns, c.empty_columns,
      c.free_column_singletons, c.bounds_tightened, c.bounds_made_finite,
      c.largest_new_finite_bound);
  std::string out(buf);
  if (!result.message.empty()) out += "message       " + result.message + "\n";
  return out;
}

}  // namespace sankhya
