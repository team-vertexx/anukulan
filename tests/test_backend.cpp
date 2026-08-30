#include <cmath>
#include <cstdio>
#include <exception>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "check.hpp"
#include "sankhya/backend.hpp"

using sankhya::BackendVector;
using sankhya::Int;
using sankhya::kInf;
using sankhya::LinAlgBackend;
using sankhya::SparseMatrix;
using sankhya::sz;
using sankhya::Triplet;

namespace {

// This file is the contract every backend has to satisfy, and it is run against
// every backend the build has - the CPU reference always, CUDA as well whenever
// the binary was built with it and a device is present.
//
// It used to say the same thing in a comment and then test `cpu_backend()`
// twice, because `b()` was a function returning the CPU backend and nothing
// took a parameter. docs/KAGGLE.md called this step "every CUDA operation
// against the CPU reference" and RESULTS.md called it the gating check; on a
// CUDA build it checked the CPU against itself and passed in a few
// milliseconds without a device ever being touched.
//
// Fixing that is not just a matter of swapping which backend `b()` returns.
// Every operation here takes backend pointers, and on CUDA those are device
// pointers: a host array passed to a kernel is not a compile error, it is a
// segfault or, worse, silence. So the vectors below go through
// allocate/upload/download, which is also how the solver uses them.

// A backend allocation with the host copies either side of it, so a test can be
// written as "these numbers in, those numbers out" on both backends.
class Vec {
 public:
  Vec(const LinAlgBackend& backend, const std::vector<double>& host)
      : backend_(&backend), v_(backend, static_cast<Int>(host.size())) {
    if (!host.empty()) v_.upload(host);
  }
  Vec(const LinAlgBackend& backend, Int n) : backend_(&backend), v_(backend, n) {}

  double* data() { return v_.data(); }
  const double* data() const { return v_.data(); }
  Int size() const { return v_.size(); }

  std::vector<double> host() const {
    std::vector<double> out;
    if (v_.size() > 0) v_.download(&out);
    return out;
  }

 private:
  const LinAlgBackend* backend_;
  BackendVector v_;
};

SparseMatrix make_matrix() {
  std::vector<Triplet> t = {
      {0, 0, 1.5}, {0, 2, -2.0}, {1, 1, 3.25}, {2, 0, 4.0}, {2, 3, -0.5}, {3, 2, 7.0},
  };
  return SparseMatrix::from_triplets(4, 4, std::move(t));
}

// Longhand y = A x, on the host, from the CSR arrays themselves.
std::vector<double> longhand_multiply(const SparseMatrix& a,
                                      const std::vector<double>& x) {
  std::vector<double> y(sz(a.rows()), 0.0);
  for (Int i = 0; i < a.rows(); ++i) {
    double sum = 0.0;
    for (Int k = a.row_begin(i); k < a.row_end(i); ++k)
      sum += a.value()[sz(k)] * x[sz(a.index()[sz(k)])];
    y[sz(i)] = sum;
  }
  return y;
}

void test_allocation_starts_at_zero(const LinAlgBackend& b) {
  // The one asymmetry that has already cost this project a wrong answer:
  // `new double[n]()` value-initialises and `cudaMalloc` does not. A missing
  // upload is invisible on the CPU and starts the device loop from whatever was
  // in the allocation. The solver relies on this for sum_x and sum_y, which are
  // accumulated into rather than written.
  const Int n = 4096;
  Vec v(b, n);
  const std::vector<double> got = v.host();
  CHECK_EQ(static_cast<Int>(got.size()), n);
  double worst = 0.0;
  for (const double d : got) worst = std::fmax(worst, std::fabs(d));
  CHECK_NEAR(worst, 0.0, 0.0);
}

void test_upload_download_fill_copy(const LinAlgBackend& b) {
  std::vector<double> host(1000);
  for (std::size_t i = 0; i < host.size(); ++i)
    host[i] = 0.5 * static_cast<double>(i) - 13.0;

  Vec v(b, host);
  const std::vector<double> back = v.host();
  for (std::size_t i = 0; i < host.size(); ++i) CHECK_NEAR(back[i], host[i], 0.0);

  Vec other(b, static_cast<Int>(host.size()));
  b.copy(v.data(), other.data(), static_cast<Int>(host.size()));
  const std::vector<double> copied = other.host();
  for (std::size_t i = 0; i < host.size(); ++i) CHECK_NEAR(copied[i], host[i], 0.0);

  // Both branches of fill: the zero one is a memset on CUDA and a kernel
  // otherwise, so they are different code paths and only one of them is the
  // one the solver leans on.
  b.fill(v.data(), static_cast<Int>(host.size()), -2.75);
  for (const double d : v.host()) CHECK_NEAR(d, -2.75, 0.0);
  b.fill(v.data(), static_cast<Int>(host.size()), 0.0);
  for (const double d : v.host()) CHECK_NEAR(d, 0.0, 0.0);
}

void test_multiply_matches_longhand(const LinAlgBackend& b) {
  const SparseMatrix a = make_matrix();
  const SparseMatrix at = a.transpose();
  const std::vector<double> x = {1.0, -2.0, 3.0, 4.0};

  Vec dx(b, x), dy(b, 4);
  b.multiply(a, dx.data(), dy.data());
  const std::vector<double> want = longhand_multiply(a, x);
  const std::vector<double> got = dy.host();
  for (Int i = 0; i < 4; ++i) CHECK_NEAR(got[sz(i)], want[sz(i)], 1e-15);

  Vec dxt(b, 4);
  b.multiply_transpose(at, dx.data(), dxt.data());
  const std::vector<double> got_t = dxt.host();
  for (Int j = 0; j < 4; ++j) {
    double w = 0.0;
    for (Int i = 0; i < 4; ++i)
      for (Int k = a.row_begin(i); k < a.row_end(i); ++k)
        if (a.index()[sz(k)] == j) w += a.value()[sz(k)] * x[sz(i)];
    CHECK_NEAR(got_t[sz(j)], w, 1e-15);
  }
}

// The CUDA product picks its lanes-per-row from the *average* row length, so
// the interesting matrices are the ones whose rows are nothing like their
// average: empty rows, a single very long row, a ragged tail. A shuffle
// reduction that reached outside its own row, or a row-to-vector mapping that
// skipped a row, would show here and nowhere in a uniform matrix.
void test_multiply_across_row_length_regimes(const LinAlgBackend& b) {
  std::mt19937 rng(20260830);
  std::uniform_real_distribution<double> pick(-3.0, 3.0);

  // (rows, cols, mean row length) chosen to land on each vector width the
  // adaptive rule can select: 2, 4, 8, 16, 32.
  const int means[] = {1, 3, 6, 12, 40};
  for (const int mean : means) {
    const Int rows = 733;  // not a multiple of the block or of any warp size
    const Int cols = 401;
    std::vector<Triplet> t;
    for (Int i = 0; i < rows; ++i) {
      // Ragged on purpose: every fifth row is empty, every hundredth is long.
      int len = mean;
      if (i % 5 == 0) len = 0;
      if (i % 100 == 7) len = 4 * mean + 37;
      for (int k = 0; k < len; ++k)
        t.push_back(Triplet{i, static_cast<Int>((i * 31 + k * 17) % cols), pick(rng)});
    }
    const SparseMatrix a = SparseMatrix::from_triplets(rows, cols, std::move(t));

    std::vector<double> x(sz(cols));
    for (double& v : x) v = pick(rng);

    Vec dx(b, x), dy(b, rows);
    // Poison the output first: a row the kernel never writes would otherwise
    // read as zero, which is also the right answer for an empty row.
    b.fill(dy.data(), rows, -12345.0);
    b.multiply(a, dx.data(), dy.data());

    const std::vector<double> want = longhand_multiply(a, x);
    const std::vector<double> got = dy.host();
    for (Int i = 0; i < rows; ++i) {
      CHECK_NEAR(got[sz(i)], want[sz(i)], 1e-13);
    }
  }
}

void test_reductions(const LinAlgBackend& b) {
  const std::vector<double> a = {1.0, -2.5, 3.0, 0.0, -7.25};
  const std::vector<double> c = {2.0, 4.0, -1.0, 9.0, 0.5};
  double want_dot = 0.0, want_sq = 0.0, want_inf = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    want_dot += a[i] * c[i];
    want_sq += a[i] * a[i];
    want_inf = std::fmax(want_inf, std::fabs(a[i]));
  }
  Vec da(b, a), dc(b, c);
  CHECK_NEAR(b.dot(da.data(), dc.data(), 5), want_dot, 1e-15);
  CHECK_NEAR(b.two_norm(da.data(), 5), std::sqrt(want_sq), 1e-15);
  CHECK_NEAR(b.inf_norm(da.data(), 5), want_inf, 0.0);

  CHECK_NEAR(b.dot(nullptr, nullptr, 0), 0.0, 0.0);
  CHECK_NEAR(b.inf_norm(nullptr, 0), 0.0, 0.0);
}

// Long enough that the CUDA reduction runs out of blocks and has to stride.
// The grid is capped at 512 blocks of 256 threads, so anything past 131,072
// entries takes the second trip round the loop - and nothing shorter tests it.
void test_reductions_past_the_block_cap(const LinAlgBackend& b) {
  const Int n = 200003;  // prime, so nothing divides evenly anywhere
  std::vector<double> a(sz(n)), c(sz(n));
  std::mt19937 rng(7);
  std::uniform_real_distribution<double> pick(-1.0, 1.0);
  for (Int i = 0; i < n; ++i) {
    a[sz(i)] = pick(rng);
    c[sz(i)] = pick(rng);
  }
  a[12345] = -9.5;  // a known maximum, past the first block's reach

  double want_dot = 0.0, want_inf = 0.0;
  for (Int i = 0; i < n; ++i) {
    want_dot += a[sz(i)] * c[sz(i)];
    want_inf = std::fmax(want_inf, std::fabs(a[sz(i)]));
  }
  Vec da(b, a), dc(b, c);
  // Summation order differs between a serial host loop and a tree of blocks,
  // so this is a relative comparison and not an exact one. 200,000 terms of
  // order one accumulate about 1e-13 of spread between orderings.
  CHECK_NEAR(b.dot(da.data(), dc.data(), n), want_dot, 1e-11);
  CHECK_NEAR(b.inf_norm(da.data(), n), want_inf, 0.0);
}

void test_primal_step_projects_and_derives(const LinAlgBackend& b) {
  const Int n = 5;
  const std::vector<double> x = {0.0, 1.0, -1.0, 5.0, 2.0};
  const std::vector<double> c = {1.0, -1.0, 0.5, 2.0, 0.0};
  const std::vector<double> kty = {0.5, 0.5, -0.5, 1.0, 3.0};
  const std::vector<double> lower = {0.0, -kInf, -2.0, 0.0, 1.0};
  const std::vector<double> upper = {1.0, 1.0, kInf, 4.0, 1.5};
  const double tau = 0.75;

  Vec dx_(b, x), dc(b, c), dkty(b, kty), dlo(b, lower), dhi(b, upper);
  Vec xn(b, n), dx(b, n), xb(b, n);
  b.primal_step(n, tau, dx_.data(), dc.data(), dkty.data(), dlo.data(), dhi.data(),
                xn.data(), dx.data(), xb.data());

  const std::vector<double> got_xn = xn.host();
  const std::vector<double> got_dx = dx.host();
  const std::vector<double> got_xb = xb.host();
  for (Int j = 0; j < n; ++j) {
    double want = x[sz(j)] - tau * (c[sz(j)] - kty[sz(j)]);
    if (want < lower[sz(j)]) want = lower[sz(j)];
    if (want > upper[sz(j)]) want = upper[sz(j)];
    CHECK_NEAR(got_xn[sz(j)], want, 0.0);
    CHECK_NEAR(got_dx[sz(j)], want - x[sz(j)], 0.0);
    CHECK_NEAR(got_xb[sz(j)], 2.0 * want - x[sz(j)], 1e-15);
    CHECK(got_xn[sz(j)] >= lower[sz(j)] - 1e-15);
    CHECK(got_xn[sz(j)] <= upper[sz(j)] + 1e-15);
  }
}

void test_dual_step_splits_at_the_equality_block(const LinAlgBackend& b) {
  const Int m = 5;
  const Int equalities = 2;
  const std::vector<double> y = {1.0, -1.0, 0.5, -0.5, 2.0};
  const std::vector<double> q = {2.0, 2.0, 2.0, 2.0, 2.0};
  const std::vector<double> kxb = {3.0, 3.0, 3.0, 3.0, 3.0};
  const std::vector<double> kx = {1.0, 1.0, 1.0, 1.0, 1.0};
  const double sigma = 0.5;

  Vec dy_(b, y), dq(b, q), dkxb(b, kxb), dkx(b, kx);
  Vec yn(b, m), dy(b, m), kdx(b, m);
  b.dual_step(m, equalities, sigma, dy_.data(), dq.data(), dkxb.data(), dkx.data(),
              yn.data(), dy.data(), kdx.data());

  const std::vector<double> got_yn = yn.host();
  const std::vector<double> got_dy = dy.host();
  const std::vector<double> got_kdx = kdx.host();
  for (Int i = 0; i < m; ++i) {
    double want = y[sz(i)] + sigma * (q[sz(i)] - kxb[sz(i)]);
    if (i >= equalities && want < 0.0) want = 0.0;
    CHECK_NEAR(got_yn[sz(i)], want, 0.0);
    CHECK_NEAR(got_dy[sz(i)], want - y[sz(i)], 0.0);
    CHECK_NEAR(got_kdx[sz(i)], 0.5 * (kxb[sz(i)] - kx[sz(i)]), 0.0);
  }
  CHECK(got_yn[1] < 0.0);
  CHECK(got_yn[3] >= 0.0);
  CHECK(got_yn[4] >= 0.0);
}

void test_kdx_identity_holds_against_a_real_matrix(const LinAlgBackend& b) {
  // k_dx is derived rather than computed, using K x_bar = 2 K x_next - K x. If
  // that identity were wrong the adaptive step size would be driven by a
  // quantity that is not K(x_next - x), and nothing would report it.
  const SparseMatrix a = make_matrix();
  const std::vector<double> x = {0.5, -1.0, 2.0, 1.0};
  const std::vector<double> x_next = {1.5, -0.5, 1.0, 2.0};
  std::vector<double> x_bar(4), diff(4);
  for (int j = 0; j < 4; ++j) {
    x_bar[sz(j)] = 2.0 * x_next[sz(j)] - x[sz(j)];
    diff[sz(j)] = x_next[sz(j)] - x[sz(j)];
  }

  Vec dx(b, x), dxb(b, x_bar), ddiff(b, diff);
  Vec k_x(b, 4), k_x_bar(b, 4), k_dx_direct(b, 4);
  b.multiply(a, dx.data(), k_x.data());
  b.multiply(a, dxb.data(), k_x_bar.data());
  b.multiply(a, ddiff.data(), k_dx_direct.data());

  const std::vector<double> zeros(4, 0.0);
  Vec zy(b, zeros), zq(b, zeros);
  Vec yn(b, 4), dy(b, 4), kdx(b, 4);
  b.dual_step(4, 4, 0.0, zy.data(), zq.data(), k_x_bar.data(), k_x.data(), yn.data(),
              dy.data(), kdx.data());

  const std::vector<double> got = kdx.host();
  const std::vector<double> want = k_dx_direct.host();
  for (int i = 0; i < 4; ++i) CHECK_NEAR(got[sz(i)], want[sz(i)], 1e-13);
}

void test_advance_kx(const LinAlgBackend& b) {
  const std::vector<double> kxb = {4.0, -2.0, 0.5};
  const std::vector<double> kx0 = {2.0, 2.0, -1.5};
  Vec dkxb(b, kxb), dkx(b, kx0);
  b.advance_kx(3, dkxb.data(), dkx.data());
  const std::vector<double> got = dkx.host();
  for (int i = 0; i < 3; ++i)
    CHECK_NEAR(got[sz(i)], 0.5 * (kxb[sz(i)] + kx0[sz(i)]), 1e-15);
}

void test_accumulate_and_average(const LinAlgBackend& b) {
  const Int n = 4;
  const std::vector<double> a = {1.0, 2.0, 3.0, 4.0};
  const std::vector<double> c = {0.5, 0.5, 0.5, 0.5};
  Vec sum(b, std::vector<double>(4, 0.0)), da(b, a), dc(b, c);
  b.accumulate(n, 2.0, da.data(), sum.data());
  b.accumulate(n, 3.0, dc.data(), sum.data());
  const std::vector<double> got = sum.host();
  for (Int j = 0; j < n; ++j)
    CHECK_NEAR(got[sz(j)], 2.0 * a[sz(j)] + 3.0 * c[sz(j)], 1e-15);

  Vec out(b, n);
  b.scale_into(n, 5.0, sum.data(), out.data());
  const std::vector<double> got_out = out.host();
  for (Int j = 0; j < n; ++j) CHECK_NEAR(got_out[sz(j)], got[sz(j)] / 5.0, 1e-15);
}

void test_blend(const LinAlgBackend& b) {
  // The Halpern update, in place. Reading z after writing it is the shape a
  // kernel gets wrong when it stages through the wrong buffer.
  const std::vector<double> z0 = {1.0, -2.0, 3.5, 0.0};
  const std::vector<double> anchor = {0.25, 0.25, -1.0, 8.0};
  Vec z(b, z0), a(b, anchor);
  b.blend(4, 0.75, z.data(), 0.25, a.data());
  const std::vector<double> got = z.host();
  for (int i = 0; i < 4; ++i)
    CHECK_NEAR(got[sz(i)], 0.75 * z0[sz(i)] + 0.25 * anchor[sz(i)], 1e-15);
}

void test_weighted_norm(const LinAlgBackend& b) {
  const std::vector<double> dx = {1.0, -2.0, 0.5};
  const std::vector<double> dy = {3.0, -1.0};
  Vec ddx(b, dx), ddy(b, dy);
  for (const double omega : {0.25, 1.0, 4.0}) {
    double want = 0.0;
    for (int j = 0; j < 3; ++j) want += omega * dx[sz(j)] * dx[sz(j)];
    for (int i = 0; i < 2; ++i) want += dy[sz(i)] * dy[sz(i)] / omega;
    CHECK_NEAR(b.weighted_norm_squared(3, 2, ddx.data(), ddy.data(), omega), want,
               1e-14);
  }
}

// Never tested before this. The fused reduction is the whole subject of one
// section of docs/KAGGLE.md and nothing checked that it returns the same two
// numbers as the three separate reductions it replaces.
void test_step_size_terms(const LinAlgBackend& b) {
  std::mt19937 rng(4242);
  std::uniform_real_distribution<double> pick(-2.0, 2.0);
  // n and m deliberately different, and both past the block cap, because the
  // fused kernel sizes one grid from max(n, m) and then walks two different
  // lengths with it.
  for (const std::pair<Int, Int> shape :
       {std::pair<Int, Int>{7, 3}, std::pair<Int, Int>{140000, 517},
        std::pair<Int, Int>{517, 140000}}) {
    const Int n = shape.first;
    const Int m = shape.second;
    std::vector<double> dx(sz(n)), dy(sz(m)), k_dx(sz(m));
    for (double& v : dx) v = pick(rng);
    for (double& v : dy) v = pick(rng);
    for (double& v : k_dx) v = pick(rng);

    Vec ddx(b, dx), ddy(b, dy), dk(b, k_dx);
    const double omega = 2.5;
    double interaction = -1.0, movement = -1.0;
    b.step_size_terms(n, m, ddx.data(), ddy.data(), dk.data(), omega, &interaction,
                      &movement);

    double want_interaction = 0.0, want_dx = 0.0, want_dy = 0.0;
    for (Int i = 0; i < m; ++i) {
      want_interaction += dy[sz(i)] * k_dx[sz(i)];
      want_dy += dy[sz(i)] * dy[sz(i)];
    }
    for (Int j = 0; j < n; ++j) want_dx += dx[sz(j)] * dx[sz(j)];
    const double want_movement = omega * want_dx + want_dy / omega;

    CHECK_NEAR(interaction, std::fabs(want_interaction), 1e-10);
    CHECK_NEAR(movement, want_movement, 1e-10);
    // It must agree with the operation it is an optimisation of, or the two
    // step-size paths take different steps on the same problem.
    CHECK_NEAR(movement,
               b.weighted_norm_squared(n, m, ddx.data(), ddy.data(), omega), 1e-10);
    // The absolute value belongs to the backend, not the caller.
    CHECK(interaction >= 0.0);
  }
}

// prepare() caches an uploaded matrix by its address and release() frees every
// one of them. Two matrices at different addresses must not share an entry, and
// a matrix used after a release must be re-uploaded rather than read from freed
// device memory. On the CPU both calls do nothing, so this whole class of bug
// is invisible there - which is exactly why it belongs in the contract.
void test_prepare_and_release(const LinAlgBackend& b) {
  const SparseMatrix a = make_matrix();
  std::vector<Triplet> t = {{0, 0, 2.0}, {1, 1, 2.0}, {2, 2, 2.0}, {3, 3, 2.0}};
  const SparseMatrix other = SparseMatrix::from_triplets(4, 4, std::move(t));

  const std::vector<double> x = {1.0, 2.0, 3.0, 4.0};
  Vec dx(b, x), y1(b, 4), y2(b, 4);

  b.prepare(a);
  b.prepare(other);
  b.multiply(a, dx.data(), y1.data());
  b.multiply(other, dx.data(), y2.data());
  const std::vector<double> want_a = longhand_multiply(a, x);
  const std::vector<double> want_o = longhand_multiply(other, x);
  std::vector<double> got_a = y1.host();
  std::vector<double> got_o = y2.host();
  for (int i = 0; i < 4; ++i) {
    CHECK_NEAR(got_a[sz(i)], want_a[sz(i)], 1e-15);
    CHECK_NEAR(got_o[sz(i)], want_o[sz(i)], 1e-15);
  }

  // Preparing twice must be idempotent, not a leak or a second upload.
  b.prepare(a);
  b.multiply(a, dx.data(), y1.data());
  got_a = y1.host();
  for (int i = 0; i < 4; ++i) CHECK_NEAR(got_a[sz(i)], want_a[sz(i)], 1e-15);

  // After a release the same matrix has to work again from scratch.
  b.release();
  b.multiply(a, dx.data(), y1.data());
  b.multiply(other, dx.data(), y2.data());
  got_a = y1.host();
  got_o = y2.host();
  for (int i = 0; i < 4; ++i) {
    CHECK_NEAR(got_a[sz(i)], want_a[sz(i)], 1e-15);
    CHECK_NEAR(got_o[sz(i)], want_o[sz(i)], 1e-15);
  }
  b.release();
}

void test_large_random_consistency(const LinAlgBackend& b) {
  std::mt19937 rng(20260825);
  std::uniform_real_distribution<double> pick(-3.0, 3.0);
  const Int n = 2000;
  std::vector<Triplet> t;
  for (Int i = 0; i < n; ++i)
    for (int k = 0; k < 5; ++k)
      t.push_back(Triplet{i, static_cast<Int>((i * 7 + k * 311) % n), pick(rng)});
  const SparseMatrix a = SparseMatrix::from_triplets(n, n, std::move(t));

  std::vector<double> x(sz(n));
  for (double& v : x) v = pick(rng);

  Vec dx(b, x), dy(b, n);
  b.multiply(a, dx.data(), dy.data());
  const std::vector<double> got = dy.host();
  const std::vector<double> want = longhand_multiply(a, x);
  for (Int i = 0; i < n; ++i) CHECK_NEAR(got[sz(i)], want[sz(i)], 1e-13);
  CHECK(std::isfinite(b.two_norm(dy.data(), n)));
}

void run_contract(const LinAlgBackend& b) {
  std::printf("  contract: %s\n", b.name().c_str());
  test_allocation_starts_at_zero(b);
  test_upload_download_fill_copy(b);
  test_multiply_matches_longhand(b);
  test_multiply_across_row_length_regimes(b);
  test_reductions(b);
  test_reductions_past_the_block_cap(b);
  test_primal_step_projects_and_derives(b);
  test_dual_step_splits_at_the_equality_block(b);
  test_kdx_identity_holds_against_a_real_matrix(b);
  test_advance_kx(b);
  test_accumulate_and_average(b);
  test_blend(b);
  test_weighted_norm(b);
  test_step_size_terms(b);
  test_prepare_and_release(b);
  test_large_random_consistency(b);
}

// Every operation, both backends, same inputs, answers compared to each other
// rather than to longhand. The tests above pin each backend to the mathematics;
// this one catches the case where both are individually defensible and the
// solver still takes a different path on the device.
void test_backends_agree(const LinAlgBackend& cpu, const LinAlgBackend& gpu) {
  std::printf("  cross-check: %s vs %s\n", cpu.name().c_str(), gpu.name().c_str());
  std::mt19937 rng(31415);
  std::uniform_real_distribution<double> pick(-2.0, 2.0);

  const Int m = 4001;
  const Int n = 2503;
  std::vector<Triplet> t;
  for (Int i = 0; i < m; ++i) {
    const int len = (i % 11 == 0) ? 0 : 2 + (i % 9);
    for (int k = 0; k < len; ++k)
      t.push_back(Triplet{i, static_cast<Int>((i * 13 + k * 101) % n), pick(rng)});
  }
  const SparseMatrix k = SparseMatrix::from_triplets(m, n, std::move(t));
  const SparseMatrix kt = k.transpose();

  std::vector<double> x(sz(n)), y(sz(m)), c(sz(n)), q(sz(m)), lo(sz(n)), hi(sz(n));
  for (Int j = 0; j < n; ++j) {
    x[sz(j)] = pick(rng);
    c[sz(j)] = pick(rng);
    lo[sz(j)] = (j % 7 == 0) ? -kInf : -2.0;
    hi[sz(j)] = (j % 5 == 0) ? kInf : 2.0;
  }
  for (Int i = 0; i < m; ++i) {
    y[sz(i)] = pick(rng);
    q[sz(i)] = pick(rng);
  }

  auto compare = [](const char* what, const std::vector<double>& a,
                    const std::vector<double>& b, double tol) {
    if (a.size() != b.size()) {
      sankhya_test::report(__FILE__, __LINE__, std::string(what) + ": size mismatch");
      return;
    }
    double worst = 0.0;
    std::size_t at = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
      const double scale = std::fmax(1.0, std::fmax(std::fabs(a[i]), std::fabs(b[i])));
      const double d = std::fabs(a[i] - b[i]) / scale;
      if (d > worst) {
        worst = d;
        at = i;
      }
    }
    if (worst > tol) {
      sankhya_test::report(__FILE__, __LINE__,
                           std::string(what) + ": worst relative difference " +
                               std::to_string(worst) + " at index " +
                               std::to_string(at));
    }
  };

  // One lambda per backend, so both run exactly the same sequence.
  struct Out {
    std::vector<double> kx, kty, x_next, dx, x_bar, y_next, dy, k_dx, kx_adv, sum,
        avg, blended;
    double dot = 0, two = 0, inf = 0, interaction = 0, movement = 0, wns = 0;
  };
  auto run = [&](const LinAlgBackend& b) {
    Out o;
    Vec dxv(b, x), dyv(b, y), dc(b, c), dq(b, q), dlo(b, lo), dhi(b, hi);
    Vec kx(b, m), kty(b, n), x_next(b, n), dxs(b, n), x_bar(b, n);
    Vec y_next(b, m), dys(b, m), k_dx(b, m), k_x_bar(b, m);

    b.multiply(k, dxv.data(), kx.data());
    b.multiply_transpose(kt, dyv.data(), kty.data());
    o.kx = kx.host();
    o.kty = kty.host();

    b.primal_step(n, 0.37, dxv.data(), dc.data(), kty.data(), dlo.data(), dhi.data(),
                  x_next.data(), dxs.data(), x_bar.data());
    o.x_next = x_next.host();
    o.dx = dxs.host();
    o.x_bar = x_bar.host();

    b.multiply(k, x_bar.data(), k_x_bar.data());
    b.dual_step(m, 137, 0.61, dyv.data(), dq.data(), k_x_bar.data(), kx.data(),
                y_next.data(), dys.data(), k_dx.data());
    o.y_next = y_next.host();
    o.dy = dys.host();
    o.k_dx = k_dx.host();

    b.advance_kx(m, k_x_bar.data(), kx.data());
    o.kx_adv = kx.host();

    o.dot = b.dot(dxv.data(), dc.data(), n);
    o.two = b.two_norm(dyv.data(), m);
    o.inf = b.inf_norm(dxv.data(), n);

    Vec sum(b, n);
    b.fill(sum.data(), n, 0.0);
    b.accumulate(n, 1.75, dxs.data(), sum.data());
    o.sum = sum.host();
    Vec avg(b, n);
    b.scale_into(n, 1.75, sum.data(), avg.data());
    o.avg = avg.host();

    Vec z(b, x), anchor(b, c);
    b.blend(n, 0.9, z.data(), 0.1, anchor.data());
    o.blended = z.host();

    b.step_size_terms(n, m, dxs.data(), dys.data(), k_dx.data(), 1.3, &o.interaction,
                      &o.movement);
    o.wns = b.weighted_norm_squared(n, m, dxs.data(), dys.data(), 1.3);
    b.release();
    return o;
  };

  const Out a = run(cpu);
  const Out g = run(gpu);

  // Elementwise results are the same arithmetic in the same order on both, so
  // they agree to rounding. The reductions sum in a different order on a GPU,
  // which is a real and permanent difference, not a defect - hence the looser
  // tolerance on those five and only those five.
  compare("K x", a.kx, g.kx, 1e-13);
  compare("K' y", a.kty, g.kty, 1e-13);
  compare("primal step x_next", a.x_next, g.x_next, 1e-14);
  compare("primal step dx", a.dx, g.dx, 1e-14);
  compare("primal step x_bar", a.x_bar, g.x_bar, 1e-14);
  compare("dual step y_next", a.y_next, g.y_next, 1e-13);
  compare("dual step dy", a.dy, g.dy, 1e-13);
  compare("dual step k_dx", a.k_dx, g.k_dx, 1e-13);
  compare("advance Kx", a.kx_adv, g.kx_adv, 1e-13);
  compare("accumulate", a.sum, g.sum, 1e-14);
  compare("scale_into", a.avg, g.avg, 1e-14);
  compare("blend", a.blended, g.blended, 1e-14);
  CHECK_NEAR(a.dot, g.dot, 1e-11);
  CHECK_NEAR(a.two, g.two, 1e-12);
  CHECK_NEAR(a.inf, g.inf, 0.0);  // a maximum has no summation order
  CHECK_NEAR(a.interaction, g.interaction, 1e-10);
  CHECK_NEAR(a.movement, g.movement, 1e-11);
  CHECK_NEAR(a.wns, g.wns, 1e-11);
}

}  // namespace

int main() {
  run_contract(sankhya::cpu_backend());

#ifdef SANKHYA_WITH_CUDA
  // Built with CUDA but possibly running where there is no device: the same
  // binary has to work on the laptop the kernels are written on and on the box
  // they are benchmarked on. A missing device is a skip; a device that fails
  // the contract is a failure.
  const LinAlgBackend* gpu = nullptr;
  try {
    gpu = &sankhya::cuda_backend();
  } catch (const std::exception& e) {
    std::printf("  skip cuda: %s\n", e.what());
  }
  if (gpu != nullptr) {
    run_contract(*gpu);
    test_backends_agree(sankhya::cpu_backend(), *gpu);
  }
#else
  std::printf("  skip cuda: built without SANKHYA_WITH_CUDA\n");
#endif

  return sankhya_test::finish("test_backend");
}
