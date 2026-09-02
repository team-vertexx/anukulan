#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

#include "check.hpp"
#include "sankhya/backend.hpp"
#include "sankhya/threading.hpp"

using sankhya::Int;
using sankhya::LinAlgBackend;
using sankhya::SparseMatrix;
using sankhya::sz;
using sankhya::Triplet;

namespace {

// The claim this file exists to hold: the threaded backend returns bit-identical
// results to the serial one, at every thread count.
//
// Bit-identical, not close. `close` to 1e-13 would pass just as happily on a
// backend that summed in a different order, and a first-order method restarts on
// a residual computed from these numbers - so a last-bit difference changes the
// iteration count, and an iteration count that moves with the thread count is
// the thing that makes a parallel solver undemonstrable. The whole design of
// threaded_backend.cpp exists to make equality the right assertion here, so
// equality is what is asserted.
//
// The thread counts include more than this machine has cores. Oversubscription
// is slow, not wrong, and a pool that only behaves when it fits is a pool that
// will misbehave on somebody else's laptop.
const std::vector<int>& thread_counts() {
  static const std::vector<int> counts = {2, 3, 4, 5, 8, 13};
  return counts;
}

bool same_bits(const std::vector<double>& a, const std::vector<double>& b) {
  if (a.size() != b.size()) return false;
  return std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0;
}

// Rows of very uneven length, which is the case a split by row count gets wrong
// and the case a real constraint matrix always has.
SparseMatrix uneven_matrix(Int rows, Int cols, unsigned seed) {
  std::mt19937 rng(seed);
  std::vector<Triplet> entries;
  for (Int r = 0; r < rows; ++r) {
    // A handful of rows hold a large share of the nonzeros.
    const Int len = (r % 97 == 0) ? cols / 4 : static_cast<Int>(rng() % 6) + 1;
    for (Int k = 0; k < len; ++k) {
      const Int c = static_cast<Int>(rng() % static_cast<unsigned>(cols));
      const double v = 1.0 + static_cast<double>(rng() % 1000) / 250.0;
      entries.push_back(Triplet{r, c, (rng() % 2) ? v : -v});
    }
  }
  return SparseMatrix::from_triplets(rows, cols, std::move(entries));
}

std::vector<double> random_vector(Int n, unsigned seed) {
  std::mt19937 rng(seed);
  std::vector<double> v(sz(n));
  for (Int i = 0; i < n; ++i) {
    v[sz(i)] = static_cast<double>(rng() % 20000) / 997.0 - 10.0;
  }
  return v;
}

void test_products_are_bit_identical() {
  const Int m = 4001, n = 2503;
  const SparseMatrix a = uneven_matrix(m, n, 12345);
  const SparseMatrix at = a.transpose();
  const std::vector<double> x = random_vector(n, 7), y = random_vector(m, 11);

  std::vector<double> want_y(sz(m)), want_x(sz(n));
  sankhya::cpu_backend().multiply(a, x.data(), want_y.data());
  sankhya::cpu_backend().multiply_transpose(at, y.data(), want_x.data());

  for (const int t : thread_counts()) {
    const LinAlgBackend& tb = sankhya::threaded_cpu_backend(t);
    std::vector<double> got_y(sz(m)), got_x(sz(n));
    tb.multiply(a, x.data(), got_y.data());
    tb.multiply_transpose(at, y.data(), got_x.data());
    CHECK(same_bits(got_y, want_y));
    CHECK(same_bits(got_x, want_x));
  }
}

void test_steps_are_bit_identical() {
  const Int n = 9001, m = 5003;
  const std::vector<double> x = random_vector(n, 3), c = random_vector(n, 5);
  const std::vector<double> kt_y = random_vector(n, 9);
  std::vector<double> lower(sz(n)), upper(sz(n));
  for (Int j = 0; j < n; ++j) {
    lower[sz(j)] = (j % 5 == 0) ? -sankhya::kInf : -2.0;
    upper[sz(j)] = (j % 7 == 0) ? sankhya::kInf : 3.0;
  }
  const std::vector<double> yv = random_vector(m, 13), q = random_vector(m, 17);
  const std::vector<double> kxb = random_vector(m, 19), kx = random_vector(m, 23);

  auto run = [&](const LinAlgBackend& b, std::vector<double>* out) {
    std::vector<double> xn(sz(n)), dx(sz(n)), xb(sz(n));
    b.primal_step(n, 0.37, x.data(), c.data(), kt_y.data(), lower.data(),
                  upper.data(), xn.data(), dx.data(), xb.data());
    std::vector<double> yn(sz(m)), dy(sz(m)), kdx(sz(m));
    b.dual_step(m, 1201, 0.61, yv.data(), q.data(), kxb.data(), kx.data(),
                yn.data(), dy.data(), kdx.data());
    std::vector<double> acc = kt_y, scaled(sz(n)), blended = x, kxc = kx;
    b.accumulate(n, 0.25, dx.data(), acc.data());
    b.scale_into(n, 1.75, acc.data(), scaled.data());
    b.blend(n, 0.4, blended.data(), 0.6, c.data());
    b.advance_kx(m, kxb.data(), kxc.data());

    out->clear();
    for (const std::vector<double>* v : {&xn, &dx, &xb, &scaled, &blended}) {
      out->insert(out->end(), v->begin(), v->end());
    }
    for (const std::vector<double>* v : {&yn, &dy, &kdx, &kxc}) {
      out->insert(out->end(), v->begin(), v->end());
    }
    out->push_back(b.inf_norm(dx.data(), n));
    out->push_back(b.inf_norm(dy.data(), m));
  };

  std::vector<double> want;
  run(sankhya::cpu_backend(), &want);
  for (const int t : thread_counts()) {
    std::vector<double> got;
    run(sankhya::threaded_cpu_backend(t), &got);
    CHECK(same_bits(got, want));
  }
}

// A loop short enough to run on the calling thread and one long enough not to
// both have to be right, because the size threshold is a branch like any other.
void test_short_and_long_loops_agree() {
  for (const Int n : {1, 2, 7, 8191, 8192, 8193, 40000}) {
    const std::vector<double> v = random_vector(n, static_cast<unsigned>(n));
    const double want = sankhya::cpu_backend().inf_norm(v.data(), n);
    for (const int t : thread_counts()) {
      CHECK(sankhya::threaded_cpu_backend(t).inf_norm(v.data(), n) == want);
    }
  }
}

// An empty matrix, an empty vector, and a matrix with empty rows: the shapes
// that a split into blocks can get wrong without any arithmetic being involved.
void test_degenerate_shapes() {
  std::vector<Triplet> entries = {{0, 0, 2.0}, {3, 1, -1.0}};
  const SparseMatrix a = SparseMatrix::from_triplets(5, 2, std::move(entries));
  const double x[2] = {1.5, -2.5};
  std::vector<double> want(5, 99.0), got(5, 99.0);
  sankhya::cpu_backend().multiply(a, x, want.data());
  for (const int t : thread_counts()) {
    std::fill(got.begin(), got.end(), 99.0);
    sankhya::threaded_cpu_backend(t).multiply(a, x, got.data());
    CHECK(same_bits(got, want));
  }
  for (const int t : thread_counts()) {
    CHECK(sankhya::threaded_cpu_backend(t).inf_norm(nullptr, 0) == 0.0);
  }
}

// One thread must hand back the serial backend itself rather than a pool of one,
// so that the default path is not paying for machinery it is not using.
void test_one_thread_is_the_serial_backend() {
  CHECK(&sankhya::threaded_cpu_backend(1) == &sankhya::cpu_backend());
  CHECK(&sankhya::threaded_cpu_backend(0) == &sankhya::cpu_backend());
}

// The split is what every claim above rests on: blocks must be in order, cover
// every row exactly once, and never overlap - overlapping blocks would have two
// threads writing one output entry, which is the race this design avoids by
// construction rather than by locking.
void test_splits_partition_exactly() {
  for (const Int rows : {0, 1, 5, 997}) {
    const SparseMatrix a = uneven_matrix(std::max<Int>(rows, 1), 51, 99);
    for (const int blocks : {1, 2, 3, 8, 64, 1000}) {
      const std::vector<Int> cut = sankhya::split_rows_by_nonzeros(a, blocks);
      CHECK(static_cast<int>(cut.size()) == blocks + 1);
      CHECK(cut.front() == 0);
      CHECK(cut.back() == a.rows());
      for (int b = 1; b <= blocks; ++b) CHECK(cut[sz(b)] >= cut[sz(b - 1)]);
    }
  }
  for (const Int n : {0, 1, 1000}) {
    for (const int blocks : {1, 3, 64}) {
      const std::vector<Int> cut = sankhya::split_range(n, blocks);
      CHECK(cut.front() == 0);
      CHECK(cut.back() == n);
      for (int b = 1; b <= blocks; ++b) CHECK(cut[sz(b)] >= cut[sz(b - 1)]);
    }
  }
}

// Every block runs exactly once, whichever thread happens to take it.
void test_pool_runs_every_block_once() {
  for (const int t : thread_counts()) {
    sankhya::ThreadPool pool(t);
    for (const int blocks : {1, 2, 7, 64, 501}) {
      std::vector<int> seen(sz(blocks), 0);
      pool.run_blocks(blocks, [&](int b) { seen[sz(b)] += 1; });
      for (const int c : seen) CHECK(c == 1);
    }
  }
}

}  // namespace

int main() {
  test_one_thread_is_the_serial_backend();
  test_splits_partition_exactly();
  test_pool_runs_every_block_once();
  test_products_are_bit_identical();
  test_steps_are_bit_identical();
  test_short_and_long_loops_agree();
  test_degenerate_shapes();
  return sankhya_test::finish("test_threading");
}
