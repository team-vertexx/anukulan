#pragma once

#include <string>
#include <vector>

#include "anukulan/sparse.hpp"

namespace anukulan {

// Sparse Cholesky factorisation, L L' = P M P', for a symmetric positive
// (semi)definite M. It exists for the interior point method, whose every
// iteration solves the normal equations
//
//     (A Theta A' + delta I) dy = r
//
// with the same sparsity pattern and a new diagonal Theta. So the work is split
// the way that pattern invites: analyse() looks at the pattern once - ordering,
// elimination tree, supernodes, where every entry of M lands in the factor -
// and factorize() is then pure arithmetic on new values, as many times as the
// method needs.
//
// Why a new factorisation rather than the two already here. lu.hpp is a
// Markowitz LU for a nonsymmetric simplex basis and would do twice the work and
// throw the symmetry away. ldl.hpp is an up-looking LDL' in the order the
// matrix arrives, and its own header records what that costs: AUG2DC does not
// finish at all in the natural order. A normal-equations matrix is worse than
// either of those cases, because every column of A turns into a dense clique
// in M, and the order decides whether the factor holds a few times M's
// nonzeros or most of a dense triangle.
//
// Three pieces, each a textbook algorithm written for this code:
//
//   ordering      approximate minimum degree on the quotient graph, with
//                 supervariables, mass elimination, aggressive absorption and
//                 dense rows set aside (Amestoy, Davis and Duff, "An
//                 approximate minimum degree ordering algorithm", SIAM J.
//                 Matrix Anal. Appl. 17(4), 1996)
//   symbolic      elimination tree by Liu's algorithm with path compression,
//                 postorder, column counts from the row subtrees, and
//                 fundamental supernodes (Liu, "The role of elimination trees
//                 in sparse factorization", SIAM J. Matrix Anal. Appl. 11(1),
//                 1990)
//   numeric       left-looking supernodal: each supernode is a dense block,
//                 updated by the supernodes below it through a relative-index
//                 scatter, then factorised in place (Ng and Peyton, "Block
//                 sparse Cholesky algorithms on advanced uniprocessor
//                 computers", SIAM J. Sci. Comput. 14(5), 1993)
//
// A pivot that comes out tiny or negative is not a failure. In the normal
// equations it is expected: rows of A that are linearly dependent, or an
// iterate so close to the optimum that the entries of Theta span twenty orders
// of magnitude, both leave a pivot that is rounding noise. Such a pivot is
// replaced by a huge number, which makes the matching component of the solution
// zero - the row is dropped from this one solve rather than divided by noise.
// This is Wright's modified Cholesky ("Modified Cholesky factorizations in
// interior-point algorithms for linear programming", SIAM J. Optim. 9(4),
// 1999), which shows the steps it produces are still good ones; the count of
// replaced pivots is reported so a caller can see it happening.
struct CholeskyOptions {
  enum class Ordering { kApproximateMinimumDegree, kNatural };
  Ordering ordering = Ordering::kApproximateMinimumDegree;

  // A pivot at or below this fraction of the column's own diagonal entry,
  // before any elimination touched it, is taken to be rounding noise. The
  // arithmetic that produced it subtracted numbers of the diagonal's size, so
  // below about machine epsilon times that size nothing of the true value is
  // left - the pivot is not small, it is unknown.
  double pivot_tolerance = 1e-13;

  // What a rejected pivot becomes. Large enough that dividing by its square
  // root sends the column's multipliers and its solution component to zero.
  double dropped_pivot = 1e128;

  // Rows with more than max(dense_row_minimum, dense_row_factor * sqrt(n))
  // off-diagonal entries are left out of the minimum degree search and ordered
  // last. A nearly dense row makes every degree update touch it, which is where
  // a minimum degree code spends its time, and it will be dense in the factor
  // whatever order the rest is eliminated in. The same rule as AMD's default.
  double dense_row_factor = 10.0;
  Int dense_row_minimum = 16;

  // Relaxed supernodes: a child supernode is merged into its parent when the
  // merged block would have at most this many columns and at most this
  // fraction of explicit zeros. Fundamental supernodes are exact but on these
  // matrices many are one or two columns wide, and a dense kernel over a
  // column or two costs more in bookkeeping than it saves.
  Int relax_columns = 16;
  double relax_zero_fraction = 0.2;
};

// Fill-reducing order for a symmetric pattern. `pattern` is square; any entry
// (i, j) or (j, i) makes i and j adjacent, and the diagonal is ignored, so
// either triangle or both may be given. Returns `perm` with perm[k] the
// original index of the k-th row eliminated.
std::vector<Int> approximate_minimum_degree(const SparseMatrix& pattern,
                                            const CholeskyOptions& options = {});

class CholeskyFactor {
 public:
  // `lower` is the LOWER triangle including the diagonal, stored row-wise: row
  // i holds the entries (i, j) with j <= i. Anything above the diagonal is
  // ignored. This is the same convention as ldl.hpp, and for the same reason
  // worth saying twice: handed the upper triangle row-wise, every off-diagonal
  // entry would be skipped and the factor would silently be diagonal.
  //
  // Only the pattern is read. `perm`, when given, is used instead of computing
  // an ordering (perm[k] = the original index eliminated k-th).
  bool analyse(const SparseMatrix& lower, const CholeskyOptions& options = {},
               std::string* error = nullptr, const std::vector<Int>* perm = nullptr);

  // Factorises values laid out exactly as the pattern given to analyse():
  // `values[e]` is the e-th stored entry of that matrix. The matrix handed to
  // the other overload must have that same pattern. Returns false only if
  // analyse() has not been called or the sizes do not match; small pivots are
  // handled as described above, not reported as failure.
  bool factorize(const std::vector<double>& values, std::string* error = nullptr);
  bool factorize(const SparseMatrix& lower, std::string* error = nullptr);

  // Solves L L' x = b in place, in the caller's ordering.
  void solve(std::vector<double>* x) const;
  void solve(double* x) const;

  Int size() const { return n_; }
  // Nonzeros of L, counting the diagonal and the explicit zeros of relaxed
  // supernodes, which is what is actually stored and computed on.
  long long nonzeros() const { return stored_nonzeros_; }
  // Nonzeros of L the pattern itself implies, without relaxation.
  long long exact_nonzeros() const { return exact_nonzeros_; }
  // Floating point operations of one factorisation, sum over columns of the
  // square of the column count. The usual way of stating it.
  double flops() const { return flops_; }
  Int supernodes() const { return static_cast<Int>(super_first_.size()) - 1; }
  Int dense_rows() const { return dense_rows_; }
  // Pivots replaced in the last factorize() because they were at or below the
  // tolerance, including negative ones.
  Int dropped_pivots() const { return dropped_pivots_; }
  // The elimination order actually used: perm()[k] is the original index of
  // the k-th pivot. It is the ordering composed with a postorder of the
  // elimination tree, which changes nothing about fill.
  const std::vector<Int>& perm() const { return perm_; }

 private:
  Int n_ = 0;
  Int input_nonzeros_ = 0;
  long long stored_nonzeros_ = 0;
  long long exact_nonzeros_ = 0;
  double flops_ = 0.0;
  Int dense_rows_ = 0;
  Int dropped_pivots_ = 0;
  double pivot_tolerance_ = 1e-13;
  double dropped_pivot_ = 1e128;
  bool factorised_ = false;

  std::vector<Int> perm_;   // new -> old
  std::vector<Int> iperm_;  // old -> new

  // Supernode s covers columns super_first_[s] .. super_first_[s+1]-1 of the
  // permuted matrix. Its rows are rows_[row_start_[s] .. row_start_[s+1]-1],
  // the first ones being its own columns, and its values are a dense
  // column-major block at value_start_[s] with that many rows.
  std::vector<Int> super_first_;
  std::vector<Int> super_of_;  // column -> supernode
  std::vector<Int> row_start_;
  std::vector<Int> rows_;
  std::vector<long long> value_start_;
  std::vector<double> values_;

  // Where each entry of the analysed lower triangle lands in values_, and the
  // permuted column each diagonal entry belongs to (-1 for off-diagonal).
  std::vector<long long> target_;
  std::vector<Int> diagonal_of_entry_;
  std::vector<double> original_diagonal_;

  // Scratch for the numeric factorisation, kept between calls.
  mutable std::vector<Int> relative_;
  std::vector<double> update_;
  std::vector<Int> link_head_;
  std::vector<Int> link_next_;
  std::vector<Int> next_row_;
  mutable std::vector<double> permuted_;
};

}  // namespace anukulan
