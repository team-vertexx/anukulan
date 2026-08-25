#pragma once

#include <string>
#include <vector>

#include "sankhya/lu.hpp"
#include "sankhya/standard_form.hpp"

namespace sankhya {

// The form the simplex actually computes in.
//
// Every row becomes an equality by giving each inequality its own slack, so the
// constraint matrix is [K | -I] over the inequality block and every constraint
// reads a'z = q. After that there are no row types left anywhere in the
// algorithm: a variable is either basic or sitting on one of its bounds, and
// that is the only case analysis the iteration needs.
struct LogicalForm {
  SparseMatrix columns;  // column-wise: row j holds column j of [K | -I]
  SparseMatrix rows;     // the same matrix row-wise
  std::vector<double> cost;
  std::vector<double> rhs;
  std::vector<double> lower;
  std::vector<double> upper;
  Int num_structural = 0;
  Int num_rows = 0;

  Int num_columns() const { return num_structural + num_rows; }
  Int num_equalities = 0;

  double objective_scale = 1.0;
  double objective_offset = 0.0;
};

LogicalForm to_logical_form(const StandardLp& lp);

// Where a non-basic variable is sitting. A basic variable's value comes from
// solving with the basis; everything else is pinned to a bound.
enum class VarStatus { kBasic, kAtLower, kAtUpper, kFree };

// The basis, its factorisation, and the quantities read off it.
class SimplexBasis {
 public:
  // Starts from the all-logical basis, which is the identity on the inequality
  // rows. Equality rows have no slack, so their basic variable has to come from
  // somewhere: the first structural column with an entry there is taken, which
  // is crude but gives a starting point that factorises.
  bool set_initial(const LogicalForm& form, std::string* error = nullptr);

  bool refactorize(const LogicalForm& form, std::string* error = nullptr);

  // x_B = B^-1 (q - N x_N), written into `values` for the basic positions and
  // the bound value for everything else.
  void compute_primal(const LogicalForm& form, std::vector<double>* values) const;

  // y = B^-T c_B, and the reduced costs d_j = c_j - y' a_j.
  void compute_duals(const LogicalForm& form, std::vector<double>* duals,
                     std::vector<double>* reduced_costs) const;

  const std::vector<Int>& basic() const { return basic_; }
  const std::vector<VarStatus>& status() const { return status_; }
  std::vector<VarStatus>& status() { return status_; }

  // Swaps `entering` in at row `leaving_row`, whose current occupant leaves to
  // the bound given. Returns false if the new basis will not factorise.
  bool pivot(const LogicalForm& form, Int leaving_row, Int entering,
             VarStatus leaving_to, std::string* error = nullptr);

  // B a_q, the pivotal column.
  void ftran_column(const LogicalForm& form, Int column,
                    std::vector<double>* out) const;

  Int updates_since_refactorization() const { return updates_; }
  const LuFactor& factors() const { return lu_; }

 private:
  std::vector<Int> basic_;
  std::vector<VarStatus> status_;
  LuFactor lu_;
  Int updates_ = 0;
};

struct SimplexOptions {
  double primal_tolerance = 1e-7;
  double dual_tolerance = 1e-7;
  // A pivot smaller than this is refused: dividing by it would wreck the
  // factorisation regardless of what the ratio test wants.
  double pivot_tolerance = 1e-7;

  Int max_iterations = 200000;
  double time_limit_seconds = 300.0;
  Int refactorization_frequency = 50;

  bool verbose = false;
  Int log_frequency = 1000;
};

enum class SimplexStatus {
  kOptimal,
  kUnbounded,
  kInfeasible,
  kIterationLimit,
  kTimeLimit,
  kNumericalError,
};

std::string to_string(SimplexStatus status);

struct SimplexResult {
  SimplexStatus status = SimplexStatus::kNumericalError;
  std::vector<double> x;  // structural variables only, in model order
  std::vector<double> y;  // row duals
  double objective = 0.0;

  Int iterations = 0;
  Int phase_one_iterations = 0;
  Int refactorizations = 0;
  double solve_seconds = 0.0;
  std::string message;
};

// Primal simplex with bounded variables. Phase one drives the basis to
// feasibility by minimising the total bound violation; phase two then optimises.
SimplexResult solve_simplex(const StandardLp& lp, const SimplexOptions& options = {});

}  // namespace sankhya
