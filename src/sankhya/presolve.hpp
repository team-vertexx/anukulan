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
//   - Complete dual postsolve. Four of the row reductions invert exactly and
//     do: an empty or redundant row cannot bind, so its dual is zero; a free
//     column singleton pins its row's dual, since the eliminated column was
//     free and needs a zero reduced cost, giving y_i = c_k / a_ik; and a
//     singleton row's dual comes from the reduced cost of the column whose
//     bound it became. Forcing and duplicate rows do not invert uniquely -
//     a forcing row's dual is only bounded by its columns' reduced costs, and a
//     duplicate's depends on which of the two rows supplied the binding bound -
//     and those get zero.
//
//     Bound tightening is the third, and it took a failing test to notice. A
//     column whose bounds were tightened can sit at the tightened bound in the
//     reduced model carrying a nonzero reduced cost, and in the original model
//     that bound does not exist, so the cost has nowhere to go. Worse, it can
//     change which reductions are available: on a two-row example, tightening
//     turned a free column into a bounded one, which made a different column
//     look implied-free, and the row dual that came back was 1 where the
//     original model requires 2. Both are right for the model they were derived
//     from.
//
//     So the recovered dual is exact when none of those three fired, and a
//     guess when they did. dual_is_exact() says which, and measure_dual_violation
//     in model.hpp says how far off - because a caller reading shadow prices off
//     a refinery plan needs to know whether they are prices or decoration.
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

  // The same column singleton when its bounds DO bind. free_column_singletons
  // above needs the column to be implied free before it will substitute, and
  // that requirement is why it almost never fires: over the three instance sets
  // here, 36,771 columns sit alone in an equality row and only 700 of them are
  // implied free. The other 36,071 were being declined.
  //
  // They do not have to be. A column alone in one row is that row's slack, and
  // the substitution is exact whether or not its bounds bind - what changes is
  // only what is left behind. For a*x_j + rest = b with x_j in [cl, cu]:
  //
  //     x_j = (b - rest) / a                        exactly, always
  //     cl <= (b - rest)/a <= cu   <=>   b - a*cu <= rest <= b - a*cl   (a > 0)
  //
  // so the column's bounds become the row's, the column goes, and the row
  // stays. This is a change of variables, not a claim that some solutions can
  // be discarded, which puts it in a different risk class from every reduction
  // that fixes a column: there is no case it can get wrong and still look
  // right, because the map between the two feasible sets is a bijection.
  //
  // free_column_singletons is the special case where the new row bounds cannot
  // bind, and it is worth more - it takes the row as well - so it is still
  // tried first and this one only picks up what it declines.
  //
  // The dual inverts exactly. Writing y for the original row dual and y' for
  // the reduced one, stationarity on any other column k of the row reads
  //     c_k - c_j*a_rk/a - sum_i A_ki y'_i  =  c_k - sum_i A_ki y_i
  // which holds for y_r = y'_r + c_j/a and y unchanged elsewhere. So this does
  // not trip dual_is_exact(), and the recovery is an accumulate rather than an
  // assignment, which is what lets it compose with a later removal of the same
  // row.
  //
  // On, measured against the same run with it off. This rule removes no rows
  // itself, so the row column below is what removing columns then exposes:
  //
  //                    rows removed      columns removed    nonzeros removed
  //   Netlib           21.58 -> 22.54     12.98 -> 20.54     16.16 -> 17.67
  //   MIPLIB            6.78 ->  6.93      4.41 ->  6.15      6.70 ->  7.18
  //   refinery          0.00 ->  0.00      0.00 -> 29.67      0.00 -> 11.90
  //
  // The refinery line is the reason this was built. Presolve removed nothing at
  // all from that model before - not one row, not one column, not one nonzero -
  // because its 1,296 equality rows leave no slack for a forcing or redundant
  // rule to find, it has no doubleton equations at all, and every one of its
  // 1,304 column singletons has a finite lower bound and so failed the
  // implied-free test. The whole of that model's reduction is this one rule.
  //
  // Netlib's row figure moves for a second reason worth separating out. The
  // implied-free test that decides between this rule and the one above was
  // rewritten to read the row's implied bounds on the column directly, which
  // also lets it fire when the row's activity is infinite in one direction and
  // that direction is the column's own. That alone - with this rule still off -
  // takes free column singletons on Netlib from 1,052 to 2,761 and rows removed
  // from 19.45% to 21.58%, because a free singleton takes its row with it. With
  // this rule on as well the free count reaches 3,385, the extra 624 being
  // singletons that only come free once a substitution has shortened their row.
  // MIPLIB is unmoved by the test alone: 6.78% either way.
  bool slack_column_singletons = true;

  // The same column singleton again, now in a row that is not an equality.
  //
  // Both rules above need an equality to stand on, because that is what pins
  // x_j to a value the objective can be rewritten around. An inequality does
  // not pin it - but a dual argument often does. x_j is in this row and nowhere
  // else, so the only two things that can stop the objective pushing it are
  // this row and its own bound in the direction of the push. If that own bound
  // is infinite, the row is the only obstacle left, and so the row is tight at
  // every optimal solution: were it slack, x_j could move another step and the
  // objective would improve. A row that is tight at every optimum can be read
  // as an equality, and then the substitution above applies unchanged.
  //
  // Which side it is tight at follows the sign of the push. Minimising, a
  // negative cost pushes x_j up and a positive one down, and the row's activity
  // follows or opposes that according to the sign of a - so it is the row's
  // upper bound that binds when those agree and the lower when they do not.
  //
  // This is the one rule here whose conclusion is about optimality rather than
  // feasibility, so it is worth being explicit that nothing is discarded: the
  // row is tight at *every* optimal solution, not merely at one of them, so the
  // whole optimal face survives. In the language of Cederberg and Boyd (2026)
  // this is weak dual exploration, and it is the second of the two fast rules
  // in their table that were not here.
  //
  // The recovered dual keeps the sign an inequality requires, which is not
  // obvious and is the reason this is safe to combine with the rule above. For
  // a <= row pushed at its upper bound with a > 0 and c_j < 0, the reduced row
  // is still a <= row, so the reduced model's own dual feasibility gives
  // y'_r <= 0, and y_r = y'_r + c_j/a <= c_j/a < 0. The mirror cases follow the
  // same way.
  //
  // On, and the measurement is the reason it is worth being explicit that this
  // is the smallest of the three changes rather than the cleverest:
  //
  //                    rows removed      columns removed    nonzeros removed
  //   Netlib           22.54 -> 23.30     20.54 -> 20.91     17.67 -> 18.73
  //   MIPLIB            6.93 ->  7.04      6.15 ->  6.30      7.18 ->  7.24
  //   refinery          0.00 ->  0.00     29.67 -> 29.67     11.90 -> 11.90
  //
  // 644 columns on Netlib and 116 on MIPLIB, and nothing at all on the refinery
  // model, whose column singletons are already in equality rows and so never
  // reach this path.
  //
  // Two things hold the count down, and both are the rule working rather than
  // failing. Dual fixing already takes the case where the objective pushes the
  // column *away* from the row, since that is a column with no lock in the
  // direction it wants to go. And an integer column cannot use this argument at
  // all - the step that proves the row tight is an arbitrarily small one, and a
  // whole number cannot take it - which is most of why MIPLIB, with 86,511
  // binary columns out of 152,174, gets so little from it.
  //
  // The static count says 3,421 column singletons sit in Netlib inequality rows
  // and 17,551 in MIPLIB ones, so 644 and 116 look like a poor conversion. They
  // are not: those two exclusions account for nearly all of the difference, and
  // what is left is the rule declining rows where the push is blocked by the
  // column's own bound instead.
  bool inequality_column_singletons = true;

  // Dual fixing. Count, for each column, the constraints that moving it in each
  // direction could break - its "locks". A column that can be pushed down
  // without breaking anything, and whose objective does not object, belongs at
  // its lower bound in some optimal solution; symmetrically for up.
  //
  // This is the empty-column rule generalised. That one fires when a column is
  // in no row at all, so nothing can stop it; this one fires whenever nothing
  // *in the direction the objective prefers* can stop it, which is a good deal
  // more often. PaPILO calls it DualFix and rates it medium cost.
  //
  // The reasoning is dual, so it costs nothing on the dual side: what it
  // concludes is that the column's reduced cost has a known sign at optimality,
  // and the row duals are untouched.
  bool dual_fixing = true;

  // A row with exactly two terms and equal bounds, a*xi + b*xj = c, lets one of
  // the two columns be written in terms of the other and substituted out,
  // taking a row and a column with it.
  //
  // Only fires when the eliminated column is implied free: the row and the
  // other column's bounds already confine it inside its own, so it has no
  // active bound and no reduced cost, which is what makes the row's dual
  // recoverable exactly. On the eighteen Netlib instances 827 of the 841
  // doubleton equations left standing by the other reductions qualify, so
  // demanding it costs almost nothing and buys an exact dual.
  //
  // This is the one reduction here that changes a coefficient of A, so it runs
  // after the others rather than among them, and the reduced matrix is
  // assembled from triplets, which merge fill without being asked.
  //
  // On, measured against the same run with it switched off. Reduction over the
  // eighteen Netlib instances with published optima:
  //
  //   rows removed      9.1%  ->  14.0%      758 doubletons
  //   columns removed  13.4%  ->  15.4%
  //   nonzeros removed  8.6%  ->  10.0%
  //
  // The nonzero line is the one worth reading twice: substitution creates fill,
  // one entry per row the eliminated column appeared in, and even so the count
  // falls - 322,824 down to 317,673. The rows it removes carry more than the
  // fill puts back.
  //
  // Over all eighty-eight instances, presolved against plain at 1e-6: 78 of 88
  // reach the published optimum against 77 without, and the geometric mean
  // speedup goes from 1.52x to 1.63x. No instance that solved before stops
  // solving.
  //
  // It runs once and does not re-enter the other reductions. A substitution can
  // leave a row that some earlier reduction would now fire on, and going back
  // for those would mean rebuilding the matrix each round to keep the activity
  // bounds honest. What that leaves on the table has not been measured.
  bool doubleton_equations = true;

  // Coefficient tightening on rows holding integer columns. Where a row can
  // only be violated when one integer column sits at its bound, that column's
  // coefficient and the row's bound can both come down by the same amount:
  //
  //   for  sum a_j x_j <= b  with a_k > 0, x_k integer, and
  //        maxact without k <= b  <  maxact without k + a_k
  //   let  d = maxact without k + a_k - b
  //   then a_k := a_k - d  and  b := b - d
  //
  // Every integer-feasible point survives - at x_k below its upper bound the
  // row was already slack and still is, and at the upper bound both sides move
  // by d - while the LP relaxation gets strictly tighter. Achterberg et al.
  // 2019, section 3.3.
  //
  // This is the second reduction here that rewrites A, so like the doubleton it
  // runs after the others and is applied when the reduced matrix is assembled.
  // It does nothing on a pure LP, since it needs an integer column to stand on.
  //
  // Off, and that is measured rather than assumed. Across the seven MIPLIB
  // instances it fires three times, all on p0201, and nowhere else. Those three
  // move p0201's root LP bound from 6875.00000004 to 6875.00000002 - which is
  // to say they move nothing, the difference being noise. A reduction that
  // fires on one instance in seven and changes no bound has not earned a place
  // in the default path, and the cost of asking is a pass over the rows.
  //
  // It is kept because it is correct and because the shape is right; what is
  // missing is coverage. As written it only handles a positive coefficient in
  // the <= reading, so the mirror case - a negative coefficient, which is the
  // same rule after reflecting the column about its bounds - is skipped
  // entirely, and it stops after the first tightening in a row because the
  // row's activity is stale by then. Fixing both would perhaps double the three
  // hits. Three that move nothing doubled is six that move nothing, which is
  // why that was not done.
  bool coefficient_tightening = false;

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
// solution. Two kinds of entry are enough on the primal side: a column either
// takes a known value, or is read off the one equality row it appeared in. That
// second form covers the doubleton substitution too - it changes coefficients
// of A, but what it records about the eliminated column is still one equality
// row solved for one unknown.
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

  // The row side, recorded as rows are removed so their duals can be put back.
  void record_zero_dual_row(Int row);
  void record_row_from_free_singleton(Int row, Int column, double coefficient);
  void record_row_from_singleton(Int row, Int column, double coefficient,
                                 double lower_created, double upper_created);
  // A doubleton equality that substituted `column` out. `column_rows` is every
  // other row the column appeared in, with its coefficient there, which is what
  // the row's own dual is recovered from.
  void record_row_from_doubleton(Int row, Int column, double coefficient,
                                 std::vector<Term> column_rows);
  // A row that kept its place but lost the column that was alone in it. The
  // row's dual moves by c_j/a and nothing else changes, so this adds to
  // whatever the row's dual already is rather than replacing it - which is what
  // makes it safe for the row to be removed by some later reduction as well.
  void record_row_from_slack_singleton(Int row, Int column, double coefficient);
  void record_unrecoverable_row(Int row);
  void set_row_dimensions(Int original_rows, std::vector<Int> reduced_row_to_original);

  Int original_rows() const { return original_rows_; }
  const std::vector<Int>& reduced_row_to_original() const {
    return reduced_row_to_original_;
  }
  // Rows whose dual could not be recovered exactly and were set to zero.
  Int unrecoverable_rows() const { return unrecoverable_rows_; }

  // Whether every reduction that fired inverts exactly on the dual side. False
  // means the vector apply_dual returns is a plausible guess, not the dual.
  bool dual_is_exact() const {
    return unrecoverable_rows_ == 0 && !bounds_were_tightened_;
  }
  void note_bound_tightening() { bounds_were_tightened_ = true; }

  // Maps the reduced model's row duals and column reduced costs back to the
  // original model's rows. `reduced_y` has reduced_rows() entries and
  // `reduced_costs` has reduced_cols(); either may be empty, in which case the
  // rows that depend on it get zero.
  std::vector<double> apply_dual(const std::vector<double>& reduced_y,
                                 const std::vector<double>& reduced_costs,
                                 const std::vector<double>& original_cost) const;

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

  // How a removed row's dual is recovered.
  enum class RowKind { kZero, kFreeSingleton, kSingleton, kDoubleton,
                       kSlackSingleton, kUnrecoverable };
  struct RowEntry {
    RowKind kind = RowKind::kZero;
    Int row = 0;
    Int column = 0;
    double coefficient = 1.0;
    double lower_created = 0.0;
    double upper_created = 0.0;
    std::vector<Term> terms;  // kDoubleton: the column's other rows
  };

  std::vector<Entry> entries_;
  std::vector<RowEntry> row_entries_;
  Int original_cols_ = 0;
  Int original_rows_ = 0;
  Int unrecoverable_rows_ = 0;
  bool bounds_were_tightened_ = false;
  std::vector<Int> reduced_to_original_;
  std::vector<Int> reduced_row_to_original_;
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
  Int slack_column_singletons = 0;
  // Of the slack singletons, those whose row was an inequality read as an
  // equality by the dual argument rather than being one already.
  Int inequality_column_singletons = 0;
  // Column singletons the substitution turned down because the row bound it
  // would have created was too large to trust. Counted rather than ignored so
  // the cap can be seen to be doing something, or seen not to be - and on
  // Netlib, MIPLIB and the refinery model it is zero, so on these three sets it
  // costs nothing and has proved nothing. It stays because the case it guards
  // against is a wrong answer rather than a missed reduction, and a guard whose
  // absence is only noticed on the instance that breaks is not worth the trade.
  Int slack_singletons_declined = 0;
  Int doubleton_equations = 0;
  Int dual_fixed_columns = 0;
  Int coefficients_tightened = 0;
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
