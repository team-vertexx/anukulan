#include <vector>

#include "check.hpp"
#include "sankhya/sparse.hpp"

using sankhya::Int;
using sankhya::Norm;
using sankhya::SparseMatrix;
using sankhya::Triplet;

namespace {

// The matrix used throughout, built out of order and with duplicates on purpose:
//   [ 1  0 -2  0 ]
//   [ 0  3  0  0 ]
//   [ 4  0  0  4 ]
SparseMatrix make_a() {
  std::vector<Triplet> t = {
      {2, 3, 5.0},   // duplicate pair, sums to 4
      {0, 2, -2.0},
      {2, 0, 4.0},
      {1, 1, 3.0},
      {2, 3, -1.0},  // ...the other half
      {0, 0, 1.0},
      {2, 1, 7.0},   // cancelling pair, must vanish entirely
      {2, 1, -7.0},
  };
  return SparseMatrix::from_triplets(3, 4, std::move(t));
}

void test_build() {
  const SparseMatrix a = make_a();
  std::string err;
  CHECK(a.validate(&err));
  if (!err.empty()) std::fprintf(stderr, "validate: %s\n", err.c_str());

  CHECK_EQ(a.rows(), 3);
  CHECK_EQ(a.cols(), 4);
  CHECK_EQ(a.nnz(), 5);  // the cancelling pair is dropped, not stored as zero

  const std::vector<Int> expect_start = {0, 2, 3, 5};
  CHECK(a.start() == expect_start);
  const std::vector<Int> expect_index = {0, 2, 1, 0, 3};
  CHECK(a.index() == expect_index);

  CHECK_NEAR(a.value()[0], 1.0, 0.0);
  CHECK_NEAR(a.value()[1], -2.0, 0.0);
  CHECK_NEAR(a.value()[2], 3.0, 0.0);
  CHECK_NEAR(a.value()[3], 4.0, 0.0);
  CHECK_NEAR(a.value()[4], 4.0, 0.0);  // 5 + (-1)
}

void test_empty_rows() {
  // A matrix whose first and last rows are empty still has to validate and
  // multiply correctly - empty rows are common after presolve.
  std::vector<Triplet> t = {{1, 0, 2.0}};
  const SparseMatrix a = SparseMatrix::from_triplets(3, 2, std::move(t));
  std::string err;
  CHECK(a.validate(&err));
  const std::vector<Int> expect_start = {0, 0, 1, 1};
  CHECK(a.start() == expect_start);

  const double x[2] = {3.0, 9.0};
  double y[3] = {-1.0, -1.0, -1.0};
  a.multiply(x, y);
  CHECK_NEAR(y[0], 0.0, 0.0);
  CHECK_NEAR(y[1], 6.0, 0.0);
  CHECK_NEAR(y[2], 0.0, 0.0);
}

void test_multiply() {
  const SparseMatrix a = make_a();
  const double x[4] = {1.0, 2.0, 3.0, 4.0};
  double y[3] = {99.0, 99.0, 99.0};
  a.multiply(x, y);
  CHECK_NEAR(y[0], -5.0, 1e-15);
  CHECK_NEAR(y[1], 6.0, 1e-15);
  CHECK_NEAR(y[2], 20.0, 1e-15);

  const double v[3] = {1.0, 2.0, 3.0};
  double w[4] = {99.0, 99.0, 99.0, 99.0};
  a.multiply_transpose(v, w);
  CHECK_NEAR(w[0], 13.0, 1e-15);
  CHECK_NEAR(w[1], 6.0, 1e-15);
  CHECK_NEAR(w[2], -2.0, 1e-15);
  CHECK_NEAR(w[3], 12.0, 1e-15);
}

void test_transpose() {
  const SparseMatrix a = make_a();
  const SparseMatrix t = a.transpose();
  std::string err;
  CHECK(t.validate(&err));
  CHECK_EQ(t.rows(), 4);
  CHECK_EQ(t.cols(), 3);
  CHECK_EQ(t.nnz(), a.nnz());

  // A^T * x computed through the explicit transpose must match multiply_transpose.
  const double v[3] = {1.0, 2.0, 3.0};
  double via_t[4] = {0, 0, 0, 0};
  double direct[4] = {0, 0, 0, 0};
  t.multiply(v, via_t);
  a.multiply_transpose(v, direct);
  for (int i = 0; i < 4; ++i) CHECK_NEAR(via_t[i], direct[i], 1e-15);

  const SparseMatrix tt = t.transpose();
  CHECK_EQ(tt.rows(), a.rows());
  CHECK_EQ(tt.cols(), a.cols());
  CHECK(tt.start() == a.start());
  CHECK(tt.index() == a.index());
  CHECK(tt.value() == a.value());
}

void test_norms() {
  const SparseMatrix a = make_a();
  double rn[3];
  a.row_norms(Norm::kInfinity, rn);
  CHECK_NEAR(rn[0], 2.0, 0.0);
  CHECK_NEAR(rn[1], 3.0, 0.0);
  CHECK_NEAR(rn[2], 4.0, 0.0);
  a.row_norms(Norm::kOne, rn);
  CHECK_NEAR(rn[0], 3.0, 0.0);
  CHECK_NEAR(rn[1], 3.0, 0.0);
  CHECK_NEAR(rn[2], 8.0, 0.0);

  double cn[4];
  a.col_norms(Norm::kInfinity, cn);
  CHECK_NEAR(cn[0], 4.0, 0.0);
  CHECK_NEAR(cn[1], 3.0, 0.0);
  CHECK_NEAR(cn[2], 2.0, 0.0);
  CHECK_NEAR(cn[3], 4.0, 0.0);
  a.col_norms(Norm::kOne, cn);
  CHECK_NEAR(cn[0], 5.0, 0.0);
  CHECK_NEAR(cn[1], 3.0, 0.0);
  CHECK_NEAR(cn[2], 2.0, 0.0);
  CHECK_NEAR(cn[3], 4.0, 0.0);

  // Column norms must agree with the row norms of the transpose.
  const SparseMatrix t = a.transpose();
  double tn[4];
  t.row_norms(Norm::kOne, tn);
  a.col_norms(Norm::kOne, cn);
  for (int i = 0; i < 4; ++i) CHECK_NEAR(tn[i], cn[i], 0.0);
}

void test_scaling() {
  SparseMatrix a = make_a();
  const double dr[3] = {2.0, 1.0, 0.5};
  a.scale_rows(dr);
  const double x[4] = {1.0, 0.0, 1.0, 1.0};
  double y[3];
  a.multiply(x, y);
  CHECK_NEAR(y[0], 2.0 * (1.0 - 2.0), 1e-15);
  CHECK_NEAR(y[1], 0.0, 1e-15);
  CHECK_NEAR(y[2], 0.5 * (4.0 + 4.0), 1e-15);

  SparseMatrix b = make_a();
  const double dc[4] = {1.0, 2.0, 3.0, 4.0};
  b.scale_cols(dc);
  double z[3];
  const double ones[4] = {1.0, 1.0, 1.0, 1.0};
  b.multiply(ones, z);
  CHECK_NEAR(z[0], 1.0 - 6.0, 1e-15);
  CHECK_NEAR(z[1], 6.0, 1e-15);
  CHECK_NEAR(z[2], 4.0 + 16.0, 1e-15);
}

void test_validate_catches_damage() {
  // validate() is only useful if it actually rejects a broken matrix, so prove it.
  SparseMatrix a = make_a();
  std::string err;
  CHECK(a.validate(&err));
  a.value().pop_back();  // now start.back() disagrees with nnz
  CHECK(!a.validate(&err));
  CHECK(!err.empty());
}

}  // namespace

int main() {
  test_build();
  test_empty_rows();
  test_multiply();
  test_transpose();
  test_norms();
  test_scaling();
  test_validate_catches_damage();
  return sankhya_test::finish("test_sparse");
}
