#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace sankhya {

using Int = std::int32_t;

inline constexpr double kInf = std::numeric_limits<double>::infinity();

// Anything at or beyond this magnitude in a model file is treated as infinite.
// MPS files in the wild use 1e30 as a stand-in for infinity.
inline constexpr double kFileInf = 1e30;

inline std::size_t sz(Int i) { return static_cast<std::size_t>(i); }

struct Triplet {
  Int row = 0;
  Int col = 0;
  double value = 0.0;
};

// kInfinity rather than kInf, because GCC's -Wshadow treats an enumerator that
// matches a namespace-scope name as shadowing it and this build is -Werror. A
// scoped enumerator cannot actually be named unqualified, so nothing was ever
// shadowed - but the first Linux build in the project's life failed on it, and
// renaming one enumerator is cheaper than arguing with the warning.
enum class Norm { kInfinity, kOne };

// Compressed sparse matrix, stored row-wise (CSR). A CSC view of A is simply the
// CSR of A^T, so there is one storage type and `transpose()` moves between them.
// Within each row the column indices are sorted and unique.
class SparseMatrix {
 public:
  SparseMatrix() = default;

  // A value that changes whenever this object's contents could have. It exists
  // because the CUDA backend caches the device-side copy of a matrix, and it
  // used to key that cache on the host address - which a stack-local matrix
  // reuses the moment the previous one goes out of scope. The backend then
  // returned the *previous* matrix's data with no error at all.
  //
  // Nothing in a solve reuses an address, so this was invisible until a test
  // built five matrices in a loop and every one after the first got the first
  // one's values back. Identity has to come from the object, not from where it
  // happens to live.
  std::uint64_t id() const { return id_; }

  // Builds from triplets. Duplicate (row, col) pairs are summed. Entries that are
  // exactly zero after summing are dropped.
  static SparseMatrix from_triplets(Int rows, Int cols, std::vector<Triplet> entries);

  Int rows() const { return rows_; }
  Int cols() const { return cols_; }
  Int nnz() const { return static_cast<Int>(value_.size()); }

  const std::vector<Int>& start() const { return start_; }
  const std::vector<Int>& index() const { return index_; }
  const std::vector<double>& value() const { return value_; }
  std::vector<double>& value() { return value_; }

  Int row_begin(Int r) const { return start_[sz(r)]; }
  Int row_end(Int r) const { return start_[sz(r) + 1]; }

  // y = A * x.  x has cols() entries, y has rows() entries. y is overwritten.
  void multiply(const double* x, double* y) const;

  // y = A^T * x.  x has rows() entries, y has cols() entries. y is overwritten.
  void multiply_transpose(const double* x, double* y) const;

  SparseMatrix transpose() const;

  // Norms of each row / each column. `out` must have rows() / cols() entries.
  // Empty rows and columns produce 0.
  void row_norms(Norm norm, double* out) const;
  void col_norms(Norm norm, double* out) const;

  // A <- diag(d) * A, with d of length rows().
  void scale_rows(const double* d);
  // A <- A * diag(d), with d of length cols().
  void scale_cols(const double* d);

  // Structural self-check. Returns false and fills `error` on the first problem.
  bool validate(std::string* error) const;

 private:
  // Assigned at construction and on every assignment, from a counter that never
  // repeats within a process.
  static std::uint64_t next_id();
  std::uint64_t id_ = next_id();

  Int rows_ = 0;
  Int cols_ = 0;
  std::vector<Int> start_{0};
  std::vector<Int> index_;
  std::vector<double> value_;
};

}  // namespace sankhya
