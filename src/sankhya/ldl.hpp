#pragma once

#include <string>
#include <vector>

#include "sankhya/sparse.hpp"

namespace sankhya {

// Sparse LDL' factorisation of a symmetric quasi-definite matrix.
//
// The matrix this exists for is the KKT system an ADMM step solves:
//
//     K = [ P + sigma I      A'      ]
//         [ A            -1/rho I    ]
//
// with P positive semidefinite, sigma > 0 and rho > 0. The leading block is
// then positive definite and the trailing one negative definite, which makes K
// quasi-definite - and Vanderbei's theorem says every symmetric permutation of
// a quasi-definite matrix has an LDL' factorisation, with D carrying the same
// inertia.
//
// That is the property the whole design rests on. It means the pivot order can
// be chosen once, from the sparsity pattern alone, and reused for every set of
// numbers - no numerical pivoting, no refactorisation when only the values
// change. A general symmetric indefinite matrix would need Bunch-Kaufman and
// would give none of that.
//
// Why this rather than the LU already in lu.hpp: that one factorises a
// nonsymmetric basis with Markowitz pivoting for stability, which is the right
// tool for a simplex basis and the wrong one here. It would do twice the work,
// store twice the factors, and throw away the symmetry that makes the ordering
// reusable.
//
// The algorithm is the up-looking LDL' of Davis, Algorithm 849: A concise
// sparse Cholesky factorization package, ACM TOMS 31(4), 2005.
//
// There is no fill-reducing ordering here, and that is a real limitation rather
// than a detail. The elimination happens in the order the matrix arrives, and
// on a structured problem that can be ruinous: Maros-Meszaros AUG2DC, 10,000
// rows by 20,200 columns, does not finish at all in the natural order, while
// conjugate gradient on the reduced system solves it in 1.19 seconds. Random
// sparse systems gave a worst fill ratio of 3.43 and told me nothing about
// that, which is what comes of measuring on the easy case.
//
// Approximate minimum degree is the fix and is not written yet. Until it is,
// predicted_nonzeros() lets a caller find out what the factor would cost before
// paying for it, and choose something else.
struct LdlOptions {
  // A pivot smaller than this in magnitude means the matrix is not
  // quasi-definite after all - usually a rho or sigma that has gone to zero.
  double minimum_pivot = 1e-14;
};

class LdlFactor {
 public:
  // `lower` is the LOWER triangle including the diagonal, stored row-wise: row
  // k holds the entries of row k whose column index is at most k. Everything
  // above the diagonal is ignored if present.
  //
  // Spelling that out because it is where the first attempt went wrong. The
  // algorithm needs, for each column k, the entries A(i, k) with i < k, and by
  // symmetry those are row k's entries with index below k. Handed the upper
  // triangle row-wise instead - where row k holds only indices at or above k -
  // every one of them is skipped, the elimination tree comes out empty, and the
  // factorisation silently becomes a diagonal one. Diagonal matrices then pass
  // their tests perfectly and nothing else does.
  //
  // analyse() reads only the pattern, so one call covers every later set of
  // values with the same structure - which is the point of quasi-definiteness.
  // `nonzero_budget` stops the analysis early once the factor is known to
  // exceed it, returning false. The tree walk costs about one step per nonzero
  // of L, so on a pattern whose fill is the problem the analysis is expensive
  // for the same reason the factorisation would be - counting all the way to
  // 2771 times the input before declining is paying most of the price of the
  // thing being declined. Zero means no budget.
  bool analyse(const SparseMatrix& lower, Int nonzero_budget = 0,
               std::string* error = nullptr);

  // Factorises values with the pattern from the last analyse(). Returns false
  // if a pivot is too small, naming the column.
  bool factorize(const SparseMatrix& lower, const LdlOptions& options = {},
                 std::string* error = nullptr);

  // Solves K x = b in place, using the factors from the last factorize().
  void solve(std::vector<double>* x) const;

  // Nonzeros the factor will hold, known from analyse() alone - before any
  // arithmetic. A caller that cannot afford the fill can find out cheaply and
  // do something else.
  Int predicted_nonzeros() const { return predicted_nonzeros_; }

  Int size() const { return n_; }
  Int nonzeros() const { return static_cast<Int>(values_.size()); }
  // Nonzeros in L over nonzeros in the upper triangle of K. The number to watch:
  // a fill ratio that climbs means the ordering is doing badly.
  double fill_ratio() const;
  // How many diagonal entries came out positive. For the KKT system above it
  // should equal the number of primal variables, and anything else means the
  // matrix was not the one this was told it would be.
  Int positive_pivots() const { return positive_pivots_; }

 private:
  Int n_ = 0;
  Int input_nonzeros_ = 0;
  Int predicted_nonzeros_ = 0;
  Int positive_pivots_ = 0;

  std::vector<Int> parent_;    // elimination tree
  std::vector<Int> column_count_;
  std::vector<Int> start_;     // L, column-wise, strictly lower triangle
  std::vector<Int> index_;
  std::vector<double> values_;
  std::vector<double> diagonal_;

  // Scratch, kept between calls so a factorisation allocates nothing.
  mutable std::vector<Int> flag_;
  mutable std::vector<Int> pattern_;
  mutable std::vector<Int> filled_;
  mutable std::vector<double> work_;
};

}  // namespace sankhya
