#pragma once

#include <string>
#include <vector>

#include "sankhya/simplex.hpp"
#include "sankhya/standard_form.hpp"

namespace sankhya {

// Turning an approximate primal-dual point into a simplex basis.
//
// The first-order method converges to a point, not to a vertex, and it gets to
// moderate accuracy quickly and to high accuracy slowly. The simplex is the
// other way round: it lands exactly on a vertex with a certificate, and its
// cost is the number of pivots it takes to walk there from wherever it started.
//
// Crossover is the join. Run the first-order method until the point is roughly
// right - which on a GPU is the cheap part - then read off which variables that
// point wants basic and hand the simplex a basis to start from instead of the
// all-logical one. If the guess is good the simplex has a short walk left, and
// what comes out is a vertex with the usual proof attached.
//
// What this is not: a full crossover in the sense of a barrier code, which also
// has to push the dual onto a face and clean up degeneracy. This produces a
// starting basis and lets the simplex do the rest. That is the cheap ninety
// percent of the idea, and it costs one FTRAN per candidate column.
//
// Why it cannot produce a singular basis, which is the trap here: it does not
// select a set of columns and hope they are independent. It starts from the
// all-logical basis, which is -I, and pivots candidates in one at a time
// through the same update the simplex uses. A pivot with a nonzero pivot
// element maps a nonsingular basis to a nonsingular basis, so the invariant
// holds at every step by construction. An earlier attempt in this codebase at
// selecting columns greedily and factorising afterwards produced singular
// starts on scfxm1, bandm and degen2; this cannot.
struct CrossoverOptions {
  // A candidate is only worth pushing if the point has it away from both of its
  // bounds. Measured relative to the variable's own range, so a column whose
  // range is 10^6 is not judged by the same absolute slack as one whose range
  // is 1.
  double interior_tolerance = 1e-6;

  // A pivot smaller than this is refused and the candidate is skipped. The
  // basis update refuses below 1e-11 anyway; this is looser on purpose, because
  // a barely-acceptable pivot here buys one column and costs every solve after
  // it.
  double pivot_tolerance = 1e-7;

  // Candidates are tried best-first. Past this many the remaining ones are the
  // ones the point was least sure about, and each still costs an FTRAN.
  // Negative means no cap.
  Int max_candidates = -1;

  // Refactorise after this many pushes. Each push is a product-form update, and
  // a thousand of them stacked without ever redoing the factorisation is the
  // same decay the simplex refactorises every hundred pivots to avoid - the
  // difference being that the simplex would have noticed, and a basis handed
  // over at the end does not get a second chance.
  //
  // cycle is what that looks like. Without this the crossover pushed 788
  // columns, handed the simplex a basis it had to roll back 51 pivots out of,
  // and the solve ended in numerical error against a cold start that reached
  // the optimum in 11,305 iterations.
  Int refactor_interval = 50;

  bool verbose = false;
};

struct CrossoverResult {
  bool ok = false;

  // What to hand the simplex: SimplexOptions::start_basic and start_status.
  std::vector<Int> basic;
  std::vector<VarStatus> status;

  // Columns the point wanted basic, and how many actually got in. The gap is
  // candidates whose pivot was too small, which is the honest measure of how
  // far the point was from a vertex.
  Int candidates = 0;
  Int pushed = 0;
  Int rejected_small_pivot = 0;
  // Rows still covered by their own logical at the end.
  Int logicals_remaining = 0;
  // The push stopped early because the basis stopped factorising.
  bool stopped_on_refactorization = false;
  // Pushes undone at the end so the basis the simplex is handed is one that
  // factorises from scratch, not one that only works through the updates.
  Int rolled_back = 0;

  std::string message;
};

// Builds a starting basis from `x`, a primal point for `lp`. The dual `y` is
// taken so a nonbasic column can be put on the bound its reduced cost prefers
// rather than the bound it happens to be nearer; pass an empty vector to skip
// that and use proximity alone.
CrossoverResult crossover_basis(const StandardLp& lp, const std::vector<double>& x,
                                const std::vector<double>& y,
                                const CrossoverOptions& options = {});

}  // namespace sankhya
