#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "sankhya/backend.hpp"
#include "sankhya/threading.hpp"

namespace sankhya {
namespace {

// How the work is cut up.
//
// More blocks than threads is what lets a worker that finished early take
// another one, and on cores of two different speeds that is the difference
// between scaling and not: on datt256_lp's K*x at six threads, a static split
// reaches 2.30x and dynamic block-stealing reaches 3.17x.
//
// But a block costs an atomic increment, and once the blocks are small enough
// that is all it is doing. On 25fv47, which has 10,400 nonzeros, the same
// comparison at six threads runs the other way: static 3.30x, dynamic 1.53x.
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
// takes less than a few microseconds serially cannot win.
//
// Swept end to end rather than on the kernels, and that distinction is the
// whole reason the number is what it is. Four instances at five threads, best
// of three, geometric mean of the solve time:
//
//     element threshold   geomean seconds
//        4,096                1.116
//        8,192                1.087
//       32,768                1.244
//      131,072                1.275
//
// Two separate pieces of kernel-level evidence pointed away from this value and
// both were wrong. bench/micro/kernel_threshold.cpp, timing the fused primal
// step back to back with a hot pool, puts the crossover nearer 4,096 - and 4,096
// measures worse here. The per-kernel profile in docs/RESULTS.md 10.4 shows the
// small kernels degrading with thread count, which argues for raising the
// threshold - and raising it is worse still. A kernel measured back to back is
// not the kernel the solver runs, and this constant is the third place in this
// file where that has now been true.
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
  // A shared_ptr rather than a reference into the ring. A reference would stay
  // valid only until the fifth distinct matrix evicted the entry under it, and
  // a split that changes shape while a loop is reading it is a race that would
  // show up as a wrong answer on some other machine, months later.
  std::shared_ptr<const std::vector<Int>> get(const SparseMatrix& matrix,
                                              int blocks) {
    const Key key{&matrix, matrix.rows(), matrix.nnz(), blocks};
    std::lock_guard<std::mutex> guard(mutex_);
    for (const Entry& e : entries_) {
      if (e.cut && e.key == key) return e.cut;
    }
    auto cut = std::make_shared<const std::vector<Int>>(
        split_rows_by_nonzeros(matrix, blocks));
    entries_[sz(static_cast<Int>(next_))] = Entry{key, cut};
    next_ = (next_ + 1) % static_cast<int>(entries_.size());
    return cut;
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
    std::shared_ptr<const std::vector<Int>> cut;
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

  int threads() const { return pool_.size(); }

  std::string name() const override {
    return "cpu-threaded(" + std::to_string(pool_.size()) + ")";
  }

  void multiply(const SparseMatrix& a, const double* x, double* y) const override {
    const Scope scope(this, "K x", threads_spmv(a));
    spmv(a, x, y);
  }

  void multiply_transpose(const SparseMatrix& at, const double* y,
                          double* x) const override {
    const Scope scope(this, "K' y", threads_spmv(at));
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
    const Scope scope(this, "fill", threads_each(n));
    each(n, [&](Int lo, Int hi) { std::fill(target + sz(lo), target + sz(hi), value); });
  }
  void copy(const double* source, double* target, Int n) const override {
    const Scope scope(this, "copy", threads_each(n));
    each(n, [&](Int lo, Int hi) {
      std::copy(source + sz(lo), source + sz(hi), target + sz(lo));
    });
  }

  // Serial, and the class comment says why.
  double dot(const double* a, const double* b, Int n) const override {
    const Scope scope(this, "dot (serial)", false);
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
    const Scope scope(this, "inf_norm", threads_each(n));
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
    const Scope scope(this, "primal_step", threads_each(n));
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
    const Scope scope(this, "dual_step", threads_each(m));
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
    const Scope scope(this, "advance_kx", threads_each(m));
    each(m, [&](Int lo, Int hi) {
      for (Int i = lo; i < hi; ++i) k_x[sz(i)] = 0.5 * (k_x_bar[sz(i)] + k_x[sz(i)]);
    });
  }

  void accumulate(Int n, double weight, const double* v, double* sum) const override {
    const Scope scope(this, "accumulate", threads_each(n));
    each(n, [&](Int lo, Int hi) {
      for (Int i = lo; i < hi; ++i) sum[sz(i)] += weight * v[sz(i)];
    });
  }

  void scale_into(Int n, double weight, const double* sum, double* out) const override {
    const Scope scope(this, "scale_into", threads_each(n));
    each(n, [&](Int lo, Int hi) {
      for (Int i = lo; i < hi; ++i) out[sz(i)] = sum[sz(i)] / weight;
    });
  }

  void blend(Int n, double a, double* z, double b, const double* anchor) const override {
    const Scope scope(this, "blend", threads_each(n));
    each(n, [&](Int lo, Int hi) {
      for (Int i = lo; i < hi; ++i) z[sz(i)] = a * z[sz(i)] + b * anchor[sz(i)];
    });
  }

  double weighted_norm_squared(Int n, Int m, const double* dx, const double* dy,
                               double omega) const override {
    return omega * dot(dx, dx, n) + dot(dy, dy, m) / omega;
  }

  // Per-kernel timing, the same facility cuda_backend.cu carries and for the
  // same reason: the question "which of these is the bottleneck" has to be
  // answered by the backend, because only it knows what it launched.
  //
  // It earns its place here rather than being a debugging aid. The loop turned
  // out to be memory-bound (docs/RESULTS.md 10.3), and once that is true the
  // thing worth knowing is which kernel is moving the most bytes - which is not
  // the one intuition picks, and was not the one this backend spent its effort
  // on first.
  //
  // A profiled run is slower than a real one, by two clock reads a call, so its
  // wall clock is not a benchmark. The proportions are what it is for.
  void set_profiling(bool on) const override {
    profiling_ = on;
    if (!on) timings_.clear();
  }

  std::string profile_report() const override {
    if (timings_.empty()) return {};
    double total = 0.0;
    long long calls = 0;
    for (const auto& [name, t] : timings_) {
      total += t.seconds;
      calls += t.calls;
    }
    std::vector<std::pair<std::string, Timing>> sorted(timings_.begin(), timings_.end());
    std::sort(sorted.begin(), sorted.end(),
              [](const auto& a, const auto& b) { return a.second.seconds > b.second.seconds; });
    char line[256];
    std::string out;
    std::snprintf(line, sizeof(line), "%-18s %11s %9s %12s %10s %10s\n", "kernel",
                  "seconds", "share", "calls", "us each", "threaded");
    out += line;
    for (const auto& [name, t] : sorted) {
      std::snprintf(line, sizeof(line), "%-18s %11.4f %8.1f%% %12lld %10.2f %10s\n",
                    name.c_str(), t.seconds,
                    total > 0.0 ? 100.0 * t.seconds / total : 0.0, t.calls,
                    t.calls > 0 ? 1e6 * t.seconds / static_cast<double>(t.calls) : 0.0,
                    t.threaded ? "yes" : "no");
      out += line;
    }
    std::snprintf(line, sizeof(line), "%-18s %11.4f %8.1f%% %12lld\n", "total", total,
                  100.0, calls);
    out += line;
    return out;
  }

 private:
  struct Timing {
    double seconds = 0.0;
    long long calls = 0;
    bool threaded = false;
  };

  // Wraps one kernel. Does nothing unless profiling is on, so the ordinary path
  // pays a branch.
  struct Scope {
    const ThreadedCpuBackend* owner;
    const char* name;
    bool threaded;
    std::chrono::steady_clock::time_point started;
    Scope(const ThreadedCpuBackend* o, const char* n, bool t)
        : owner(o), name(n), threaded(t) {
      if (owner->profiling_) started = std::chrono::steady_clock::now();
    }
    ~Scope() {
      if (!owner->profiling_) return;
      const double el = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - started).count();
      Timing& t = owner->timings_[name];
      t.seconds += el;
      t.calls += 1;
      t.threaded = threaded;
    }
  };

  // Whether a call of this size actually spreads, so the report can say which
  // kernels are running on one thread despite the pool existing. That column is
  // the point of the table: a kernel below the threshold is serial no matter
  // what --threads says.
  bool threads_each(Int n) const {
    return n >= kMinElementsToThread && pool_.size() > 1;
  }
  bool threads_spmv(const SparseMatrix& a) const {
    return a.nnz() >= kMinNonzerosToThread && pool_.size() > 1;
  }

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
    const std::shared_ptr<const std::vector<Int>> held = splits_.get(a, blocks);
    const std::vector<Int>& cut = *held;
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
  mutable bool profiling_ = false;
  mutable std::map<std::string, Timing> timings_;
};

std::mutex g_backend_mutex;
// Kept per thread count and never replaced, so a reference handed out for one
// solve cannot be destroyed by a later call asking for a different count. A
// process asks for one count in practice; the cost of the general case is a
// pool that outlives its use, which is threads sitting on a condition variable.
std::vector<std::unique_ptr<ThreadedCpuBackend>> g_threaded;

}  // namespace

const LinAlgBackend& threaded_cpu_backend(int threads) {
  if (threads <= 1) return cpu_backend();
  std::lock_guard<std::mutex> guard(g_backend_mutex);
  for (const std::unique_ptr<ThreadedCpuBackend>& b : g_threaded) {
    if (b->threads() == threads) return *b;
  }
  g_threaded.push_back(std::make_unique<ThreadedCpuBackend>(threads));
  return *g_threaded.back();
}

}  // namespace sankhya
