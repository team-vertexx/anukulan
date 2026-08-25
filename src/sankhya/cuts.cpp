#include "sankhya/cuts.hpp"

#include <algorithm>
#include <cmath>

namespace sankhya {
namespace {

double fractional_part(double v) { return v - std::floor(v); }

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
    double slack;  // 1 - x*, how far from one the relaxation left it
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
      items.push_back(Item{j, -coefficient, x[sz(j)]});
    } else {
      items.push_back(Item{j, coefficient, 1.0 - x[sz(j)]});
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
    const bool complemented = items[idx].slack == x[sz(j)];
    if (complemented) {
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
  return acceptable(*out, options);
}

}  // namespace

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
      if (options.cover_cuts &&
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

  std::sort(cuts.begin(), cuts.end(),
            [](const Cut& a, const Cut& b) { return a.violation > b.violation; });
  if (static_cast<Int>(cuts.size()) > options.max_cuts)
    cuts.resize(sz(options.max_cuts));
  return cuts;
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

}  // namespace sankhya
