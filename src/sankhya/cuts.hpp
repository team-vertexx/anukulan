#pragma once

#include <vector>

#include "sankhya/simplex.hpp"
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
  // Rows past this index are cuts added in an earlier round rather than rows of
  // the model. Zero means "no limit", which is what this did before.
  //
  // Separating a cover from a previously generated cut is where fiber lost its
  // answer: in round 8 a cover taken off an earlier cut row produced an
  // inequality the optimum violates by 1.0, and the solver then proved a wrong
  // optimum with a matching dual bound.
  Int separate_only_before_row = 0;

  bool cover_cuts = true;
  bool mir_cuts = true;
  bool gomory_cuts = true;

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

// Gomory mixed-integer cuts, read off a row of the simplex tableau.
//
// These were absent because there was no tableau to read: a first-order method
// has no basis. The dual simplex that solves branch and bound's nodes has one,
// and the row it needs is the same B^-T e_r the dual's ratio test already
// computes.
//
// `basic` and `status` are a basis the simplex ended on, as returned in
// SimplexResult. A row generates a cut when its basic variable is an integer
// column sitting at a fractional value. Returns nothing rather than failing if
// the basis will not factorise.
//
// Unlike the cover and MIR separators, these are not derived from the model's
// own rows but from a combination of them, so a sign error produces an
// inequality that looks entirely reasonable and quietly removes solutions.
// test_cuts.cpp enumerates every feasible integer point of its fixtures and
// checks that none is cut off; that is the only thing that catches it.
// Rank by efficacy and keep a near-orthogonal subset, so that cuts from
// different families compete on one list rather than each getting a quota.
std::vector<Cut> select_cuts(std::vector<Cut> cuts, const CutOptions& options = {});

std::vector<Cut> separate_gomory_cuts(const StandardLp& lp,
                                      const std::vector<bool>& integral,
                                      const std::vector<Int>& basic,
                                      const std::vector<VarStatus>& status,
                                      const CutOptions& options = {});

}  // namespace sankhya
