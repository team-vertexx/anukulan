#pragma once

#include <string>
#include <vector>

#include "sankhya/model.hpp"

namespace sankhya {

// Presolve: shrink the model before it ever reaches a solver, then map the
// reduced solution back.
//
// This is the one part of a solver where the win comes from doing nothing
// clever. Real models are written by people and by modelling languages, and
// they carry rows that cannot bind, columns that are already fixed, and
// constraints that are really just bounds wearing a row's clothes. Every one of
// those costs a first-order method iterations and a simplex pivots, for no
// information. Andersen and Andersen (Math. Prog. 71, 1995) and, for the
// integer case, Achterberg et al. (INFORMS J. Comput. 32(2), 2020) are the two
// papers this follows.
//
// What is deliberately not here:
//
//   - Doubleton equations. Eliminating x from a two-term equality substitutes
//     into every row the other variable touches, which changes coefficients of
//     A and can turn a sparse matrix dense. Every other reduction below leaves
//     A's nonzeros exactly as they were and only edits bounds, costs and the
//     alive flags, which is what makes postsolve a short replay instead of a
//     second matrix build. That property was worth keeping.
//   - Dual reductions (dominated and weakly dominated columns). They are sound
//     for an LP but they fix variables using an argument about optimality
//     rather than feasibility, so they can cut off alternative optima. That is
//     fine when only the objective is wanted and wrong when the point itself
//     is the deliverable - which, for a refinery plan, it is.
//   - Dual postsolve. What comes back is a primal point, exact and feasible for
//     the original model. Row duals of the reduced model do not map back
//     without recording more than this stack records, so the LP dual is
//     reported from an unpresolved solve. Stated here rather than discovered
//     later.
struct PresolveOptions {
  bool empty_rows = true;
  bool singleton_rows = true;      // a row with one term is a bound
  bool redundant_rows = true;      // activity bounds already inside the row bounds
  bool forcing_rows = true;        // activity can only be met at one end
  bool duplicate_rows = true;      // parallel rows merge into the tighter one
  bool fixed_columns = true;       // lower == upper, substitute it out
  bool empty_columns = true;       // in no row, so the objective decides it
  bool free_column_singletons = true;  // in one equality row, bounds not binding
  bool bound_tightening = true;    // interval arithmetic on each row

  Int max_rounds = 30;

  // How close a reduction has to be to exact before it is allowed to fire.
  double feasibility_tolerance = 1e-9;

  // How large a violation has to be before presolve is willing to say the word
  // "infeasible". Deliberately far looser than the tolerance above, and they are
  // two numbers rather than one for a reason worth writing down.
  //
  // Substituting a fixed column out of a row moves that row's bounds, and a
  // model where a few thousand columns are fixed accumulates rounding in every
  // row they touched. On Netlib's maros that accumulation reaches 4e-08 - which
  // is nothing, and which a single tolerance of 1e-9 read as proof that a
  // perfectly feasible model was infeasible. Anything between the two numbers is
  // treated as "do not reduce", never as a conclusion. Presolve is allowed to
  // decline; it is not allowed to be confidently wrong.
  double infeasibility_tolerance = 1e-7;

  // Implied bounds are exact consequences of a row, so they are applied as
  // computed. An earlier version pushed them outward by 1e-9 to avoid shaving
  // the optimum, which was seven orders of magnitude more slack than the
  // arithmetic actually loses - and that manufactured slack then fed the forcing
  // rule below, which pinned whole rows of variables onto bounds that were only
  // there because of the padding. Zero is the right default; the knob stays so
  // the effect can be measured rather than argued about.
  double bound_relaxation = 0.0;

  // An implied bound only replaces an infinite one when it is smaller than this.
  // A bound of 1e9 is not information, it is a number large enough to be
  // arithmetic noise, and real presolvers cap this for the same reason.
  //
  // Worth recording what this did NOT fix, because it was written to fix it.
  // On pilot87 presolve makes the first-order method need far more iterations,
  // and the signature is distinctive: the dual residual falls to 3e-10 while the
  // duality gap sits at 5e-03. The guess was that tightening turns thousands of
  // infinite bounds into large finite ones, and that a large bound against a
  // small reduced cost leaves a gap term that will not close. Capping them at
  // 1e7 moved the gap from 5.67e-03 to 6.84e-03 - which is to say the guess was
  // wrong, and the cap is kept on its own merits rather than as a fix.
  //
  // What the measurement did settle: over eighteen Netlib instances, bound
  // tightening on is 1.33x geomean against 1.21x with it off, with zero wrong
  // answers either way, and it turns maros from an iteration-limit failure into
  // an optimal solve in 261,280 iterations. pilot87 is one instance out of
  // eighteen, and --presolve-no-bound-tightening exists for it.
  double max_new_finite_bound = 1e7;

  bool verbose = false;
};

enum class PresolveStatus {
  kReduced,     // a reduced model is in `reduced`
  kInfeasible,  // proved infeasible without solving anything
  kUnbounded,   // an empty column can run to an infinite bound
};

std::string to_string(PresolveStatus status);

// The record of everything removed, replayed in reverse to rebuild a full
// solution. Two kinds of entry are enough because no reduction here changes a
// coefficient of A: a column either takes a known value, or is read off the one
// equality row it appeared in.
class PostsolveStack {
 public:
  struct Term {
    Int col = 0;
    double coefficient = 0.0;
  };

  void record_fixed(Int col, double value);
  void record_singleton(Int col, double coefficient, double rhs,
                        std::vector<Term> other_terms);

  void set_dimensions(Int original_cols, std::vector<Int> reduced_to_original);

  Int original_cols() const { return original_cols_; }
  Int reduced_cols() const { return static_cast<Int>(reduced_to_original_.size()); }
  const std::vector<Int>& reduced_to_original() const { return reduced_to_original_; }

  // Scatters the reduced solution into the original column order and replays the
  // removals in reverse. `reduced_x` must have reduced_cols() entries.
  std::vector<double> apply(const std::vector<double>& reduced_x) const;

 private:
  enum class Kind { kFixed, kSingleton };
  struct Entry {
    Kind kind = Kind::kFixed;
    Int col = 0;
    double value = 0.0;        // kFixed: the value. kSingleton: the row's rhs.
    double coefficient = 1.0;  // kSingleton: this column's coefficient.
    std::vector<Term> terms;   // kSingleton: the rest of the row.
  };

  std::vector<Entry> entries_;
  Int original_cols_ = 0;
  std::vector<Int> reduced_to_original_;
};

struct PresolveCounts {
  Int rows_removed = 0;
  Int cols_removed = 0;
  Int nonzeros_removed = 0;

  Int empty_rows = 0;
  Int singleton_rows = 0;
  Int redundant_rows = 0;
  Int forcing_rows = 0;
  Int duplicate_rows = 0;
  Int fixed_columns = 0;
  Int empty_columns = 0;
  Int free_column_singletons = 0;
  Int bounds_tightened = 0;
  Int bounds_made_finite = 0;   // was infinite, now is not
  double largest_new_finite_bound = 0.0;
  Int rounds = 0;
};

struct PresolveResult {
  PresolveStatus status = PresolveStatus::kReduced;
  Model reduced;
  PostsolveStack postsolve;
  PresolveCounts counts;
  double seconds = 0.0;
  std::string message;

  // The original model's row and column counts, so a report can show both ends.
  Int original_rows = 0;
  Int original_cols = 0;
  Int original_nnz = 0;
};

// Column-removing reductions are skipped when the model has a Hessian: fixing or
// substituting a column out of a quadratic objective rewrites Q's neighbours as
// well as c, and getting that subtly wrong would be worse than not doing it. Row
// reductions and bound tightening still run, and those are the ones a QP gets
// its size back from anyway.
PresolveResult presolve(const Model& model, const PresolveOptions& options = {});

std::string format_presolve(const PresolveResult& result);

}  // namespace sankhya
