#include "sankhya/cuts.hpp"

#include <algorithm>
#include <cmath>

namespace sankhya {
namespace {

double fractional_part(double v) { return v - std::floor(v); }

// Column indices are kept sorted so the pairwise cosine below can walk two cuts
// together instead of building a dense vector for each comparison.
void sort_by_column(Cut* cut) {
  std::vector<std::size_t> order(cut->columns.size());
  for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
  std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
    return cut->columns[a] < cut->columns[b];
  });
  std::vector<Int> columns(cut->columns.size());
  std::vector<double> coefficients(cut->coefficients.size());
  for (std::size_t i = 0; i < order.size(); ++i) {
    columns[i] = cut->columns[order[i]];
    coefficients[i] = cut->coefficients[order[i]];
  }
  cut->columns.swap(columns);
  cut->coefficients.swap(coefficients);
}

// Rejects a cut that is numerically wide, or that has collapsed to nothing.
bool acceptable(const Cut& cut, const CutOptions& options) {
  if (cut.columns.empty()) return false;
  if (cut.violation < options.min_violation) return false;
  double lo = kInf;
  double hi = 0.0;
  for (const double c : cut.coefficients) {
    const double a = std::fabs(c);
    if (a < 1e-12) continue;
    lo = std::fmin(lo, a);
    hi = std::fmax(hi, a);
  }
  if (hi == 0.0) return false;
  return hi / lo <= options.max_dynamic_range;
}

// Mixed integer rounding on a single row, with the two steps that make it
// actually fire: complementation and scaling.
//
// Plain MIR only produces something when the right-hand side has a non-zero
// fractional part, and on most real rows it does not. Marchand and Wolsey's
// c-MIR procedure fixes that with two moves, and skipping either one is why a
// first attempt at this separated cuts on one instance out of seven:
//
//   Complementation. A variable sitting near its upper bound in the current
//   point is substituted as x_j = u_j - x'_j, so the interesting variables are
//   the ones near zero in the transformed row.
//
//   Scaling. The row is multiplied by delta before rounding. Trying delta = 1
//   and delta = 1/|a_j| for each integer coefficient costs almost nothing and is
//   what turns an integral right-hand side into a fractional one.
//
// Written as sum_j a_j x_j - s <= b over non-negative integers x and continuous
// s, with f = b - floor(b) and f_j = a_j - floor(a_j), the MIR inequality is
//
//   sum_j [ floor(a_j) + max(0, (f_j - f)/(1-f)) ] x_j - s/(1-f) <= floor(b)
struct Substitution {
  Int column;
  double coefficient;   // in the transformed, non-negative variable
  bool complemented;
  bool integral;
};

bool mir_with_scale(const StandardLp& lp, const std::vector<Substitution>& terms,
                    double base_rhs, const std::vector<double>& x, double delta,
                    const CutOptions& options, Cut* out) {
  if (!(std::fabs(delta) > 1e-12)) return false;
  const double b = base_rhs / delta;
  const double f = fractional_part(b);
  if (f < 1e-4 || f > 1.0 - 1e-4) return false;

  out->columns.clear();
  out->coefficients.clear();
  double activity = 0.0;
  double rhs_shift = 0.0;

  for (const Substitution& term : terms) {
    const double coefficient = term.coefficient / delta;
    double cut_coefficient = 0.0;
    if (term.integral) {
      const double fj = fractional_part(coefficient);
      cut_coefficient =
          std::floor(coefficient) + std::fmax(0.0, (fj - f) / (1.0 - f));
    } else {
      cut_coefficient = std::fmin(0.0, coefficient) / (1.0 - f);
    }
    if (std::fabs(cut_coefficient) < 1e-12) continue;

    // Value of the transformed variable at the current point.
    const std::size_t j = sz(term.column);
    const double transformed =
        term.complemented ? lp.upper[j] - x[j] : x[j] - lp.lower[j];
    activity += cut_coefficient * transformed;

    // Map the coefficient back onto the original variable.
    if (term.complemented) {
      out->columns.push_back(term.column);
      out->coefficients.push_back(-cut_coefficient);
      rhs_shift -= cut_coefficient * lp.upper[j];
    } else {
      out->columns.push_back(term.column);
      out->coefficients.push_back(cut_coefficient);
      rhs_shift += cut_coefficient * lp.lower[j];
    }
  }

  const double cut_rhs = std::floor(b);
  out->violation = activity - cut_rhs;
  if (out->violation < options.min_violation) return false;

  // Negate into the >= convention the standard form uses.
  for (double& c : out->coefficients) c = -c;
  out->rhs = -(cut_rhs + rhs_shift);
  out->family = "mir";
  sort_by_column(out);
  return acceptable(*out, options);
}

bool mir_from_row(const StandardLp& lp, const std::vector<bool>& integral,
                  const std::vector<double>& x, Int row, double scale,
                  const CutOptions& options, Cut* out) {
  std::vector<Substitution> terms;
  double b = -scale * lp.q[sz(row)];
  bool any_integer = false;

  for (Int e = lp.k.row_begin(row); e < lp.k.row_end(row); ++e) {
    const Int j = lp.k.index()[sz(e)];
    const double coefficient = -scale * lp.k.value()[sz(e)];
    if (std::fabs(coefficient) < 1e-12) continue;
    const std::size_t sj = sz(j);
    const double lower = lp.lower[sj];
    const double upper = lp.upper[sj];
    const bool is_int = integral[sj];

    // Complement when the point sits in the upper half of the range, so the
    // transformed variable is the one near zero.
    bool complement = false;
    if (upper < kInf && lower > -kInf) {
      const double middle = 0.5 * (lower + upper);
      complement = x[sj] > middle;
    } else if (lower <= -kInf && upper < kInf) {
      complement = true;  // no finite lower bound; the upper one has to serve
    } else if (lower <= -kInf) {
      return false;       // free variable, no non-negative substitution exists
    }

    if (complement) {
      b -= coefficient * upper;
      terms.push_back(Substitution{j, -coefficient, true, is_int});
    } else {
      b -= coefficient * lower;
      terms.push_back(Substitution{j, coefficient, false, is_int});
    }
    if (is_int) any_integer = true;
  }
  if (!any_integer) return false;

  // Candidate scalings: the row as it stands, and one that makes each integer
  // coefficient unit size.
  std::vector<double> deltas{1.0};
  for (const Substitution& term : terms) {
    if (!term.integral) continue;
    const double a = std::fabs(term.coefficient);
    if (a > 1e-9 && a < 1e9) deltas.push_back(a);
    if (deltas.size() > 12) break;
  }

  Cut best;
  bool found = false;
  Cut candidate;
  for (const double delta : deltas) {
    if (mir_with_scale(lp, terms, b, x, delta, options, &candidate)) {
      if (!found || candidate.violation > best.violation) {
        best = candidate;
        found = true;
      }
    }
  }
  if (found) *out = best;
  return found;
}

// Cover inequality from a row that behaves like a binary knapsack.
//
// With sum_{j in C} a_j > b over binaries, at least one member of C must be
// zero, so sum_{j in C} x_j <= |C| - 1. Separation is the greedy one: take the
// items the relaxation likes most, cheapest per unit of weight, until the
// capacity is exceeded.
bool cover_from_row(const StandardLp& lp, const std::vector<bool>& integral,
                    const std::vector<double>& x, Int row, double scale,
                    const CutOptions& options, Cut* out) {
  struct Item {
    Int column;
    double weight;
    double slack;  // how far this item is from the bound that would fill it
    // Whether x_j was replaced by 1 - x_j to make its weight positive.
    //
    // Recorded rather than worked out again later. It used to be recovered by
    // testing slack == x_j, which is true for a complemented item and false for
    // an uncomplemented one - except at x_j = 0.5, where 1 - x_j is also 0.5
    // and the test says complemented about everything. A simplex vertex can sit
    // a binary at exactly 0.5, and then the cut came out with +1 where it
    // needed -1 and removed feasible points. Separating only at random interior
    // points never landed on 0.5 and never saw it.
    bool complemented;
  };
  std::vector<Item> items;
  double capacity = -scale * lp.q[sz(row)];

  for (Int e = lp.k.row_begin(row); e < lp.k.row_end(row); ++e) {
    const Int j = lp.k.index()[sz(e)];
    const double coefficient = -scale * lp.k.value()[sz(e)];
    if (std::fabs(coefficient) < 1e-12) continue;
    const bool binary = integral[sz(j)] && lp.lower[sz(j)] == 0.0 &&
                        lp.upper[sz(j)] == 1.0;
    if (!binary) return false;  // only pure binary knapsacks here
    if (coefficient < 0.0) {
      // Complement: x_j -> 1 - x_j turns a negative weight positive.
      capacity -= coefficient;
      items.push_back(Item{j, -coefficient, x[sz(j)], true});
    } else {
      items.push_back(Item{j, coefficient, 1.0 - x[sz(j)], false});
    }
  }
  if (items.size() < 2 || capacity <= 0.0) return false;

  std::sort(items.begin(), items.end(), [](const Item& p, const Item& q) {
    return p.slack / std::fmax(1e-12, p.weight) < q.slack / std::fmax(1e-12, q.weight);
  });

  double weight = 0.0;
  double slack_sum = 0.0;
  std::size_t taken = 0;
  for (; taken < items.size() && weight <= capacity; ++taken) {
    weight += items[taken].weight;
    slack_sum += items[taken].slack;
  }
  if (weight <= capacity) return false;  // no cover exists in this row

  // Violated exactly when the members' slacks sum to less than one.
  out->violation = 1.0 - slack_sum;
  if (out->violation < options.min_violation) return false;

  // sum_{j in C} x_j <= |C| - 1, written as a >= row after negation, with
  // complemented members folded back.
  out->columns.clear();
  out->coefficients.clear();
  double rhs = -(static_cast<double>(taken) - 1.0);
  for (std::size_t idx = 0; idx < taken; ++idx) {
    const Int j = items[idx].column;
    if (items[idx].complemented) {
      out->columns.push_back(j);
      out->coefficients.push_back(1.0);
      rhs += 1.0;
    } else {
      out->columns.push_back(j);
      out->coefficients.push_back(-1.0);
    }
  }
  out->rhs = rhs;
  out->family = "cover";
  sort_by_column(out);
  return acceptable(*out, options);
}

}  // namespace

std::vector<Cut> select_cuts(std::vector<Cut> cuts, const CutOptions& options) {
  // Score by efficacy, then keep a near-orthogonal subset. This is the standard
  // selection from the branch-and-cut literature (Wesselmann and Suhl; it is
  // also what SCIP's default selector does): take the best remaining cut, drop
  // everything too parallel to it, repeat.
  for (Cut& cut : cuts) {
    double norm_squared = 0.0;
    for (const double c : cut.coefficients) norm_squared += c * c;
    cut.norm = std::sqrt(norm_squared);
    cut.efficacy = cut.norm > 1e-12 ? cut.violation / cut.norm : 0.0;
  }
  std::sort(cuts.begin(), cuts.end(),
            [](const Cut& a, const Cut& b) { return a.efficacy > b.efficacy; });

  auto cosine = [](const Cut& a, const Cut& b) {
    if (a.norm <= 1e-12 || b.norm <= 1e-12) return 0.0;
    // Sparse dot product over the shared columns.
    std::size_t ia = 0;
    std::size_t ib = 0;
    double dot = 0.0;
    while (ia < a.columns.size() && ib < b.columns.size()) {
      if (a.columns[ia] == b.columns[ib]) {
        dot += a.coefficients[ia] * b.coefficients[ib];
        ++ia;
        ++ib;
      } else if (a.columns[ia] < b.columns[ib]) {
        ++ia;
      } else {
        ++ib;
      }
    }
    return std::fabs(dot) / (a.norm * b.norm);
  };

  std::vector<Cut> selected;
  for (const Cut& next : cuts) {
    if (static_cast<Int>(selected.size()) >= options.max_cuts_per_round) break;
    bool duplicate = false;
    for (const Cut& kept : selected) {
      if (cosine(next, kept) > options.max_parallelism) {
        duplicate = true;
        break;
      }
    }
    if (!duplicate) selected.push_back(next);
  }
  return selected;
}

std::vector<Cut> separate_cuts(const StandardLp& lp, const std::vector<bool>& integral,
                               const std::vector<double>& x,
                               const CutOptions& options) {
  std::vector<Cut> cuts;
  Cut candidate;

  for (Int row = 0; row < lp.num_rows() && static_cast<Int>(cuts.size()) < options.max_cuts;
       ++row) {
    // A >= row read directly, and an equality read from both sides.
    const bool equality = row < lp.num_equalities;
    for (const double scale : (equality ? std::vector<double>{1.0, -1.0}
                                        : std::vector<double>{1.0})) {
      const bool original_row = options.separate_only_before_row <= 0 ||
                                row < options.separate_only_before_row;
      if (options.cover_cuts && original_row &&
          cover_from_row(lp, integral, x, row, scale, options, &candidate)) {
        cuts.push_back(candidate);
        continue;  // a cover from this row is stronger than a MIR from it
      }
      if (options.mir_cuts &&
          mir_from_row(lp, integral, x, row, scale, options, &candidate)) {
        cuts.push_back(candidate);
      }
    }
  }

  return select_cuts(std::move(cuts), options);
}

StandardLp append_cuts(const StandardLp& lp, const std::vector<Cut>& cuts) {
  if (cuts.empty()) return lp;

  StandardLp out;
  out.c = lp.c;
  out.lower = lp.lower;
  out.upper = lp.upper;
  out.num_equalities = lp.num_equalities;
  out.objective_scale = lp.objective_scale;
  out.objective_offset = lp.objective_offset;

  const Int extra = static_cast<Int>(cuts.size());
  const Int rows = lp.num_rows() + extra;
  out.q.resize(sz(rows));
  out.row_origin.resize(sz(rows));

  std::vector<Triplet> entries;
  entries.reserve(sz(lp.k.nnz()) + cuts.size() * 8);
  for (Int i = 0; i < lp.num_rows(); ++i) {
    out.q[sz(i)] = lp.q[sz(i)];
    out.row_origin[sz(i)] = lp.row_origin[sz(i)];
    for (Int e = lp.k.row_begin(i); e < lp.k.row_end(i); ++e) {
      entries.push_back(Triplet{i, lp.k.index()[sz(e)], lp.k.value()[sz(e)]});
    }
  }
  for (std::size_t c = 0; c < cuts.size(); ++c) {
    const Int dest = lp.num_rows() + static_cast<Int>(c);
    out.q[sz(dest)] = cuts[c].rhs;
    // A cut has no model row behind it; -1 marks that so duals are never mapped
    // back to a constraint the user wrote.
    out.row_origin[sz(dest)] = StandardLp::RowOrigin{-1, 1.0};
    for (std::size_t idx = 0; idx < cuts[c].columns.size(); ++idx) {
      entries.push_back(
          Triplet{dest, cuts[c].columns[idx], cuts[c].coefficients[idx]});
    }
  }
  out.k = SparseMatrix::from_triplets(rows, lp.num_cols(), std::move(entries));
  out.kt = out.k.transpose();
  return out;
}

std::vector<Cut> separate_gomory_cuts(const StandardLp& lp,
                                      const std::vector<bool>& integral,
                                      const std::vector<Int>& basic,
                                      const std::vector<VarStatus>& status,
                                      const CutOptions& options) {
  std::vector<Cut> cuts;
  const LogicalForm form = to_logical_form(lp);
  const Int structural = form.num_structural;
  const Int columns = form.columns.rows();
  if (static_cast<Int>(basic.size()) != form.num_rows) return cuts;

  SimplexBasis b;
  std::string error;
  if (!b.set_from(form, basic, status, &error)) return cuts;

  std::vector<double> x;
  b.compute_primal(form, &x);

  // A logical variable is continuous whatever its row was, and a structural one
  // only counts as integer here if the bound it is sitting on is a whole
  // number - the displacement from that bound is what has to be integral.
  auto integral_displacement = [&](Int j, VarStatus st) {
    if (j >= structural) return false;
    if (!integral[sz(j)]) return false;
    const double bound = st == VarStatus::kAtUpper ? form.upper[sz(j)]
                                                   : form.lower[sz(j)];
    return std::isfinite(bound) && std::fabs(bound - std::round(bound)) < 1e-9;
  };
  std::vector<double> rho;
  std::vector<double> alpha(sz(columns), 0.0);
  std::vector<double> coefficient(sz(columns), 0.0);
  std::vector<double> logical_part(sz(form.num_rows), 0.0);
  std::vector<double> structural_part(sz(structural), 0.0);

  for (Int r = 0; r < form.num_rows && static_cast<Int>(cuts.size()) < options.max_cuts;
       ++r) {
    const Int leaving = basic[sz(r)];
    if (leaving >= structural || !integral[sz(leaving)]) continue;
    const double value = x[sz(leaving)];
    const double f0 = value - std::floor(value);
    // Too close to integral and the cut is nearly trivial; too close to 1 or 0
    // and dividing by f0 or 1 - f0 turns rounding into coefficients.
    if (f0 < 1e-6 || f0 > 1.0 - 1e-6) continue;

    b.pivot_row(form, r, &rho);

    // The row of the tableau, and the sign that turns each nonbasic into a
    // non-negative displacement from the bound it is sitting on.
    bool usable = true;
    for (Int j = 0; j < columns; ++j) {
      alpha[sz(j)] = 0.0;
      const VarStatus st = status[sz(j)];
      if (st == VarStatus::kBasic) continue;
      double dot = 0.0;
      for (Int e = form.columns.row_begin(j); e < form.columns.row_end(j); ++e)
        dot += form.columns.value()[sz(e)] * rho[sz(form.columns.index()[sz(e)])];
      if (std::fabs(dot) < 1e-11) continue;
      if (st == VarStatus::kFree) {
        // A free nonbasic has no bound to measure from, so its displacement can
        // go either way and the derivation below does not hold. Skip the row
        // rather than produce something that looks like a cut.
        usable = false;
        break;
      }
      alpha[sz(j)] = st == VarStatus::kAtUpper ? -dot : dot;
    }
    if (!usable) continue;

    std::fill(coefficient.begin(), coefficient.end(), 0.0);
    double largest = 0.0;
    double smallest = kInf;
    for (Int j = 0; j < columns; ++j) {
      const double a = alpha[sz(j)];
      if (a == 0.0) continue;
      double c = 0.0;
      if (integral_displacement(j, status[sz(j)])) {
        const double fj = a - std::floor(a);
        c = (fj <= f0) ? fj / f0 : (1.0 - fj) / (1.0 - f0);
      } else {
        c = (a > 0.0) ? a / f0 : -a / (1.0 - f0);
      }
      if (c <= 0.0) continue;
      coefficient[sz(j)] = c;
      largest = std::fmax(largest, c);
      smallest = std::fmin(smallest, c);
    }
    if (largest <= 0.0) continue;
    // A cut whose coefficients span too many orders of magnitude is numerical
    // noise wearing an inequality's clothes.
    if (largest / smallest > options.max_dynamic_range) continue;

    // Back to x. A displacement at a lower bound is x_j - l_j and at an upper
    // bound is u_j - x_j, so the sign on x_j follows the bound, and each bound
    // moves the right hand side.
    std::fill(structural_part.begin(), structural_part.end(), 0.0);
    std::fill(logical_part.begin(), logical_part.end(), 0.0);
    double rhs = 1.0;
    for (Int j = 0; j < columns; ++j) {
      const double c = coefficient[sz(j)];
      if (c == 0.0) continue;
      const bool at_upper = status[sz(j)] == VarStatus::kAtUpper;
      const double bound = at_upper ? form.upper[sz(j)] : form.lower[sz(j)];
      if (!std::isfinite(bound)) { rhs = kInf; break; }
      rhs += at_upper ? -c * bound : c * bound;
      const double signed_c = at_upper ? -c : c;
      if (j < structural) structural_part[sz(j)] += signed_c;
      else logical_part[sz(j - structural)] += signed_c;
    }
    if (!std::isfinite(rhs)) continue;

    // The logicals are slacks: s = K x - q, so their share of the cut folds
    // into the structural coefficients and the right hand side.
    for (Int i = 0; i < form.num_rows; ++i) {
      const double d = logical_part[sz(i)];
      if (d == 0.0) continue;
      rhs += d * lp.q[sz(i)];
      for (Int e = lp.k.row_begin(i); e < lp.k.row_end(i); ++e)
        structural_part[sz(lp.k.index()[sz(e)])] += d * lp.k.value()[sz(e)];
    }

    Cut cut;
    cut.family = "gomory";
    double norm_squared = 0.0;
    double activity = 0.0;
    for (Int j = 0; j < structural; ++j) {
      const double c = structural_part[sz(j)];
      if (std::fabs(c) < 1e-11) continue;
      cut.columns.push_back(j);
      cut.coefficients.push_back(c);
      norm_squared += c * c;
      activity += c * x[sz(j)];
    }
    if (cut.columns.empty() || norm_squared <= 0.0) continue;
    cut.rhs = rhs;
    cut.norm = std::sqrt(norm_squared);
    cut.violation = rhs - activity;
    cut.efficacy = cut.violation / cut.norm;
    if (cut.violation < options.min_violation) continue;
    cuts.push_back(std::move(cut));
  }
  return cuts;
}

}  // namespace sankhya
