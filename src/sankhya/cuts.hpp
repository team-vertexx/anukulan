#pragma once

#include <vector>

#include "sankhya/standard_form.hpp"

namespace sankhya {

// Cutting planes generated from the original constraint rows, without a simplex
// tableau.
//
// The obvious cut family for a branch-and-cut code is Gomory mixed-integer,
// read off a row of the simplex tableau. A first-order method has no tableau and
// no basis, so that door is shut until the simplex exists.
//
// It is not the only door. Marchand and Wolsey (Aggregation and Mixed Integer
// Rounding to Solve MIPs, Operations Research 49(3), 2001) showed that MIR
// inequalities can be separated from the original rows, or simple aggregations
// of them, and that this is strong enough to be worth doing on its own - SCIP's
// default aggregation separator still works this way. Cover inequalities are
// likewise a property of a single row plus the variable bounds.
//
// So both families below need only the constraint matrix and the current
// relaxation point.
struct Cut {
  // sum_j coefficient_j * x_j >= rhs, matching the standard form's row sense.
  std::vector<Int> columns;
  std::vector<double> coefficients;
  double rhs = 0.0;
  double violation = 0.0;   // how far the current point is on the wrong side
  double efficacy = 0.0;    // violation divided by the cut's norm
  double norm = 0.0;
  const char* family = "";
};

struct CutOptions {
  Int max_cuts = 200;
  double min_violation = 1e-5;
  // Cuts whose largest and smallest coefficients differ by more than this are
  // discarded. A numerically wide cut buys a little tightening and costs a lot
  // of conditioning, which a first-order method feels more than simplex does.
  double max_dynamic_range = 1e6;
  bool cover_cuts = true;
  bool mir_cuts = true;

  // Cuts are ranked by efficacy - violation divided by the cut's own norm -
  // rather than by raw violation, which just rewards large coefficients.
  //
  // Then a near-parallel cut is dropped in favour of the better one already
  // selected. Two cuts pointing the same way tighten the relaxation once but
  // cost two rows in every node below, and a first-order method pays for extra
  // rows in every single iteration. Dumping every violated cut into the problem
  // made p0201 go from finding the exact optimum to missing it by 9%.
  double max_parallelism = 0.9;   // cosine similarity above this is a duplicate
  Int max_cuts_per_round = 20;
};

// Separates violated cuts at `x`. Only rows whose integer structure supports a
// cut are considered; everything else is skipped cheaply.
std::vector<Cut> separate_cuts(const StandardLp& lp, const std::vector<bool>& integral,
                               const std::vector<double>& x,
                               const CutOptions& options = {});

// Appends cuts to the constraint matrix, returning the enlarged problem. Cuts
// are >= rows, so they join the inequality block.
StandardLp append_cuts(const StandardLp& lp, const std::vector<Cut>& cuts);

}  // namespace sankhya
