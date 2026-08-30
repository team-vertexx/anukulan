#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <mutex>

#include "sankhya/backend.hpp"
#include "sankhya/threading.hpp"

namespace sankhya {
namespace {

// How the work is cut up.
//
// More blocks than threads is what lets a worker that finished early take
// another one, and on cores of two different speeds that is the difference
// between scaling and not: on datt256_lp's K*x at six threads, one block per
// thread reaches 2.42x and eight blocks per thread reaches 3.15x.
//
// But a block costs an atomic increment, and once the blocks are small enough
// that is all it is doing. 25fv47 has 10,400 nonzeros, so eight blocks a thread
// at seven threads is 186 nonzeros a block, and there dynamic scheduling
// measures 0.36x where a plain one-block-per-thread split measures 3.40x.
//
// So the block count follows the work rather than the thread count: enough
// nonzeros in a block to pay for taking it, and never more than eight blocks a
// thread nor fewer than one.
constexpr int kMaxBlocksPerThread = 8;
constexpr Int kMinNonzerosPerBlock = 4096;
constexpr Int kMinElementsPerBlock = 8192;

int blocks_for(Int work, Int min_per_block, int threads) {
  if (threads <= 1) return 1;
  const Int by_work = work / std::max<Int>(min_per_block, 1);
  const int wanted = static_cast<int>(std::min<Int>(by_work, threads * kMaxBlocksPerThread));
  return std::max(threads, std::min(wanted, threads * kMaxBlocksPerThread));
}

// Below these sizes the loop is run on the calling thread. A barrier costs
// about half a microsecond at the thread counts this pool uses, so a loop that
// takes less than a few microseconds serially cannot win. The crossover is
// measured in docs/RESULTS.md.
constexpr Int kMinElementsToThread = 8192;
constexpr Int kMinNonzerosToThread = 32768;

// Row splits, cached because recomputing one costs a pass over the rows and the
// first-order loop multiplies by the same two matrices every iteration.
//
// A ring rather than a map keyed by address with an explicit release(), which
// is the shape the CUDA backend uses: branch and bound builds a fresh problem
// at every node, so anything keyed by address and freed only on request grows
// without bound. Four entries covers K and K' for a solve and the same pair for
// a nested polishing sub-solve, and anything past that evicts itself.
class SplitCache {
 public:
  const std::vector<Int>& get(const SparseMatrix& matrix, int blocks) {
    const Key key{&matrix, matrix.rows(), matrix.nnz(), blocks};
    std::lock_guard<std::mutex> guard(mutex_);
    for (const Entry& e : entries_) {
      if (e.key == key) return e.cut;
    }
    Entry fresh{key, split_rows_by_nonzeros(matrix, blocks)};
    entries_[sz(static_cast<Int>(next_))] = std::move(fresh);
    const std::size_t placed = sz(static_cast<Int>(next_));
    next_ = (next_ + 1) % static_cast<int>(entries_.size());
    return entries_[placed].cut;
  }

 private:
  struct Key {
    const SparseMatrix* matrix = nullptr;
    Int rows = -1;
    Int nnz = -1;
    int blocks = 0;
    bool operator==(const Key& o) const {
      return matrix == o.matrix && rows == o.rows && nnz == o.nnz &&
             blocks == o.blocks;
    }
  };
  struct Entry {
    Key key;
    std::vector<Int> cut;
  };
  std::array<Entry, 4> entries_{};
  int next_ = 0;
  std::mutex mutex_;
};

// The CPU backend with the elementwise work and the matrix products spread over
// a thread pool.
//
// The determinism claim, which is the whole reason this is a separate class
// rather than a flag inside the serial one:
//
//   Every operation below produces bit-identical output to cpu_backend(), at
//   every thread count, on every run.
//
// It holds because of what is and is not threaded. A matrix product splits by
// rows, and each output entry is still accumulated by exactly the serial inner
// loop in exactly the serial order - only the assignment of rows to threads
// varies, and no two blocks write the same entry. The elementwise steps have no
// cross-entry dependence at all. inf_norm is a maximum, and maximum is
// associative and commutative in floating point, so any partition gives the
// same answer.
//
// dot is the exception and it is deliberately left serial. Floating point
// addition is not associative, so any split changes the sum in the last bits -
// and a first-order method restarts on a residual computed from these sums, so
// the last bits decide the iteration count. Threading it would buy whatever
// share the inner products hold and cost the guarantee; the share is measured
// in docs/RESULTS.md and it is small.
class ThreadedCpuBackend final : public LinAlgBackend {
 public:
  explicit ThreadedCpuBackend(int threads) : pool_(threads) {}

  std::string name() const override {
    return "cpu-threaded(" + std::to_string(pool_.size()) + ")";
  }

  void multiply(const SparseMatrix& a, const double* x, double* y) const override {
    spmv(a, x, y);
  }

  void multiply_transpose(const SparseMatrix& at, const double* y,
                          double* x) const override {
    spmv(at, y, x);
  }

  double* allocate(Int n) const override {
    return n > 0 ? new double[sz(n)]() : nullptr;
  }
  void deallocate(double* p) const override { delete[] p; }
  void upload(const double* host, double* target, Int n) const override {
    std::copy(host, host + sz(n), target);
  }
  void download(const double* source, double* host, Int n) const override {
    std::copy(source, source + sz(n), host);
  }
  void fill(double* target, Int n, double value) const override {
    each(n, [&](Int lo, Int hi) { std::fill(target + sz(lo), target + sz(hi), value); });
  }
  void copy(const double* source, double* target, Int n) const override {
    each(n, [&](Int lo, Int hi) {
      std::copy(source + sz(lo), source + sz(hi), target + sz(lo));
    });
  }

  // Serial, and the class comment says why.
  double dot(const double* a, const double* b, Int n) const override {
    double sum = 0.0;
    for (Int i = 0; i < n; ++i) sum += a[sz(i)] * b[sz(i)];
    return sum;
  }

  double two_norm(const double* a, Int n) const override {
    return std::sqrt(dot(a, a, n));
  }

  // Maximum is exactly associative, so this one splits without changing the
  // answer.
  double inf_norm(const double* a, Int n) const override {
    if (n < kMinElementsToThread || pool_.size() == 1) {
      double m = 0.0;
      for (Int i = 0; i < n; ++i) m = std::fmax(m, std::fabs(a[sz(i)]));
      return m;
    }
    const int blocks = blocks_for(n, kMinElementsPerBlock, pool_.size());
    const std::vector<Int> cut = split_range(n, blocks);
    std::vector<double> partial(sz(static_cast<Int>(blocks)), 0.0);
    pool_.run_blocks(blocks, [&](int b) {
      double m = 0.0;
      for (Int i = cut[sz(static_cast<Int>(b))]; i < cut[sz(static_cast<Int>(b) + 1)]; ++i) {
        m = std::fmax(m, std::fabs(a[sz(i)]));
      }
      partial[sz(static_cast<Int>(b))] = m;
    });
    double m = 0.0;
    for (const double p : partial) m = std::fmax(m, p);
    return m;
  }

  void primal_step(Int n, double tau, const double* x, const double* c,
                   const double* kt_y, const double* lower, const double* upper,
                   double* x_next, double* dx, double* x_bar) const override {
    each(n, [&](Int lo, Int hi) {
      for (Int j = lo; j < hi; ++j) {
        const std::size_t sj = sz(j);
        double v = x[sj] - tau * (c[sj] - kt_y[sj]);
        if (v < lower[sj]) v = lower[sj];
        if (v > upper[sj]) v = upper[sj];
        x_next[sj] = v;
        const double d = v - x[sj];
        dx[sj] = d;
        x_bar[sj] = v + d;
      }
    });
  }

  void dual_step(Int m, Int num_equalities, double sigma, const double* y,
                 const double* q, const double* k_x_bar, const double* k_x,
                 double* y_next, double* dy, double* k_dx) const override {
    each(m, [&](Int lo, Int hi) {
      for (Int i = lo; i < hi; ++i) {
        const std::size_t si = sz(i);
        double v = y[si] + sigma * (q[si] - k_x_bar[si]);
        if (i >= num_equalities && v < 0.0) v = 0.0;
        y_next[si] = v;
        dy[si] = v - y[si];
        k_dx[si] = 0.5 * (k_x_bar[si] - k_x[si]);
      }
    });
  }

  void advance_kx(Int m, const double* k_x_bar, double* k_x) const override {
    each(m, [&](Int lo, Int hi) {
      for (Int i = lo; i < hi; ++i) k_x[sz(i)] = 0.5 * (k_x_bar[sz(i)] + k_x[sz(i)]);
    });
  }

  void accumulate(Int n, double weight, const double* v, double* sum) const override {
    each(n, [&](Int lo, Int hi) {
      for (Int i = lo; i < hi; ++i) sum[sz(i)] += weight * v[sz(i)];
    });
  }

  void scale_into(Int n, double weight, const double* sum, double* out) const override {
    each(n, [&](Int lo, Int hi) {
      for (Int i = lo; i < hi; ++i) out[sz(i)] = sum[sz(i)] / weight;
    });
  }

  void blend(Int n, double a, double* z, double b, const double* anchor) const override {
    each(n, [&](Int lo, Int hi) {
      for (Int i = lo; i < hi; ++i) z[sz(i)] = a * z[sz(i)] + b * anchor[sz(i)];
    });
  }

  double weighted_norm_squared(Int n, Int m, const double* dx, const double* dy,
                               double omega) const override {
    return omega * dot(dx, dx, n) + dot(dy, dy, m) / omega;
  }

 private:
  // One elementwise loop over [0, n), given as a range so the body keeps the
  // serial inner loop verbatim.
  template <typename Body>
  void each(Int n, Body&& body) const {
    if (n <= 0) return;
    if (n < kMinElementsToThread || pool_.size() == 1) {
      body(Int{0}, n);
      return;
    }
    const int blocks = blocks_for(n, kMinElementsPerBlock, pool_.size());
    const std::vector<Int> cut = split_range(n, blocks);
    pool_.run_blocks(blocks, [&](int b) {
      body(cut[sz(static_cast<Int>(b))], cut[sz(static_cast<Int>(b) + 1)]);
    });
  }

  void spmv(const SparseMatrix& a, const double* in, double* out) const {
    const Int rows = a.rows();
    if (rows <= 0) return;
    if (a.nnz() < kMinNonzerosToThread || pool_.size() == 1) {
      a.multiply(in, out);
      return;
    }
    const int blocks = blocks_for(a.nnz(), kMinNonzerosPerBlock, pool_.size());
    const std::vector<Int>& cut = splits_.get(a, blocks);
    const auto& start = a.start();
    const auto& index = a.index();
    const auto& value = a.value();
    pool_.run_blocks(blocks, [&](int b) {
      for (Int r = cut[sz(static_cast<Int>(b))]; r < cut[sz(static_cast<Int>(b) + 1)]; ++r) {
        double sum = 0.0;
        for (Int k = start[sz(r)]; k < start[sz(r) + 1]; ++k) {
          sum += value[sz(k)] * in[index[sz(k)]];
        }
        out[r] = sum;
      }
    });
  }

  mutable ThreadPool pool_;
  mutable SplitCache splits_;
};

std::mutex g_backend_mutex;
std::unique_ptr<ThreadedCpuBackend> g_threaded;
int g_threaded_size = 0;

}  // namespace

const LinAlgBackend& threaded_cpu_backend(int threads) {
  if (threads <= 1) return cpu_backend();
  std::lock_guard<std::mutex> guard(g_backend_mutex);
  if (!g_threaded || g_threaded_size != threads) {
    g_threaded = std::make_unique<ThreadedCpuBackend>(threads);
    g_threaded_size = threads;
  }
  return *g_threaded;
}

}  // namespace sankhya
