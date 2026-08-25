#pragma once

#include <string>
#include <vector>

#include "sankhya/sparse.hpp"

namespace sankhya {

// Sparse LU factorisation of a simplex basis, with the two operations the
// revised simplex is built from.
//
// A basis matrix from a linear program is not a general sparse matrix. It is
// mostly triangular already - slack columns are unit vectors, and structural
// columns tend to chain - so the first thing any serious factorisation does is
// peel off the part that needs no elimination at all. Singleton rows and columns
// are removed repeatedly until none is left, and only the residue, the nucleus,
// gets Markowitz elimination. On typical LP bases the nucleus is a small
// fraction of the matrix, which is why this is fast enough to redo every fifty
// iterations.
//
// Following Suhl and Suhl, Computing Sparse LU Factorizations for Large-Scale
// Linear Programming Bases, ORSA Journal on Computing 2(1), 1990.
struct LuOptions {
  // Markowitz pivots must also be numerically acceptable: a candidate is
  // rejected unless its magnitude is at least this fraction of the largest
  // entry in its column. 0.1 is the usual compromise between sparsity and
  // stability - closer to 1 is stabler and denser.
  double stability_threshold = 0.1;
  // Entries smaller than this are treated as structural zeros.
  double drop_tolerance = 1e-12;
};

class LuFactor {
 public:
  // Factorises the basis whose columns are `basis` taken from `columns`, which
  // must be the constraint matrix stored column-wise. Returns false if the basis
  // is singular, with `error` describing which row or column failed.
  bool factorize(const SparseMatrix& columns, const std::vector<Int>& basis,
                 const LuOptions& options = {}, std::string* error = nullptr);

  // Solves B x = b in place. This is FTRAN in simplex terminology.
  void ftran(std::vector<double>* x) const;

  // Solves B' x = b in place. This is BTRAN.
  void btran(std::vector<double>* x) const;

  Int size() const { return n_; }
  Int nonzeros() const;
  // Nonzeros in the factors divided by nonzeros in the basis. The number to
  // watch: when it grows the factorisation is decaying and wants redoing.
  double fill_ratio() const;
  Int triangular_prefix() const { return triangular_; }
  Int nucleus_size() const { return n_ - triangular_; }

 private:
  struct Eta {
    Int pivot_row = 0;
    std::vector<Int> rows;
    std::vector<double> values;
    double pivot = 1.0;
  };

  Int n_ = 0;
  Int basis_nonzeros_ = 0;
  Int triangular_ = 0;

  // Elimination order, and where each pivot sat in the original matrix.
  std::vector<Int> pivot_row_;     // step -> row
  std::vector<Int> pivot_column_;  // step -> column
  std::vector<Eta> etas_;          // L, as elementary transformations

  // U, stored by column in elimination order.
  std::vector<std::vector<Int>> u_rows_;
  std::vector<std::vector<double>> u_values_;
  std::vector<double> u_diagonal_;

  // Row and column to step, for the permutations.
  std::vector<Int> row_step_;
  std::vector<Int> column_step_;
};

}  // namespace sankhya
