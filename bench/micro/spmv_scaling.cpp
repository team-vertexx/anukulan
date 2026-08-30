// Throwaway harness for one question: how far does the first-order method's
// inner loop scale on this machine, and what actually stops it?
//
// Three things are measured separately because they have different answers:
//   - a streaming triad, which is this machine's memory ceiling
//   - K*x and K'*y under a static nonzero-balanced split
//   - the same under dynamic block scheduling, which is what heterogeneous
//     cores need
//   - the bare barrier, which is the floor under any per-iteration threading
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <thread>
#include <vector>

#include "sankhya/mps_reader.hpp"
#include "sankhya/scaling.hpp"
#include "sankhya/standard_form.hpp"

using namespace sankhya;
using Clock = std::chrono::steady_clock;

// Bounded spin then yield. Pure spinning is what made a ten-thread barrier cost
// six milliseconds: ten spinners on ten cores with anything else on the box
// means the OS deschedules a spinner and everyone waits for its quantum.
struct Pool {
  int threads;
  std::vector<std::thread> workers;
  std::atomic<unsigned> generation{0};
  std::atomic<int> arrived{0};
  std::atomic<bool> stop{false};
  const std::function<void(int)>* job = nullptr;

  explicit Pool(int t) : threads(t) {
    for (int i = 1; i < threads; ++i) {
      workers.emplace_back([this, i] {
        unsigned seen = 0;
        for (;;) {
          int spins = 0;
          while (generation.load(std::memory_order_acquire) == seen) {
            if (stop.load(std::memory_order_relaxed)) return;
            if (++spins > 4000) { std::this_thread::yield(); spins = 3000; }
          }
          seen = generation.load(std::memory_order_acquire);
          if (stop.load(std::memory_order_relaxed)) return;
          (*job)(i);
          arrived.fetch_add(1, std::memory_order_release);
        }
      });
    }
  }
  void run(const std::function<void(int)>& f) {
    if (threads == 1) { f(0); return; }
    job = &f;
    arrived.store(0, std::memory_order_relaxed);
    generation.fetch_add(1, std::memory_order_release);
    f(0);
    int spins = 0;
    while (arrived.load(std::memory_order_acquire) != threads - 1) {
      if (++spins > 4000) { std::this_thread::yield(); spins = 3000; }
    }
  }
  ~Pool() {
    stop.store(true, std::memory_order_relaxed);
    generation.fetch_add(1, std::memory_order_release);
    for (auto& w : workers) w.join();
  }
};

static void spmv_rows(const SparseMatrix& A, const double* in, double* out, Int lo, Int hi) {
  const auto& st = A.start(); const auto& ix = A.index(); const auto& va = A.value();
  for (Int r = lo; r < hi; ++r) {
    double s = 0.0;
    for (Int k = st[sz(r)]; k < st[sz(r) + 1]; ++k) s += va[sz(k)] * in[ix[sz(k)]];
    out[r] = s;
  }
}

// Row boundaries for `parts` chunks of roughly equal nonzero count.
static std::vector<Int> split_by_nnz(const SparseMatrix& A, int parts) {
  std::vector<Int> c(sz(parts) + 1, 0);
  const Int nz = A.nnz(), R = A.rows();
  Int at = 0;
  for (int i = 1; i < parts; ++i) {
    const Int target = static_cast<Int>((static_cast<long long>(nz) * i) / parts);
    while (at < R && A.row_end(at) < target) ++at;
    c[sz(i)] = at;
  }
  c[sz(parts)] = R;
  for (int i = 1; i <= parts; ++i) if (c[sz(i)] < c[sz(i - 1)]) c[sz(i)] = c[sz(i - 1)];
  return c;
}

template <typename F>
static double timed(int reps, F&& f) {
  auto t0 = Clock::now();
  for (int i = 0; i < reps; ++i) f();
  return std::chrono::duration<double>(Clock::now() - t0).count() / reps;
}

int main(int argc, char** argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: spmv_scaling <file.mps> [maxthreads]\n"); return 2; }
  const int maxt = argc > 2 ? std::atoi(argv[2]) : 10;

  auto read = read_mps(argv[1]);
  if (!read.ok) { std::fprintf(stderr, "read failed: %s\n", read.error.c_str()); return 1; }
  auto sf = to_standard_form(read.model);
  StandardLp lp = sf.lp;
  scale_lp(&lp);

  const SparseMatrix& K = lp.k;
  const SparseMatrix& Kt = lp.kt;
  const Int m = K.rows(), n = K.cols();
  std::printf("# %s  m=%d n=%d nnz=%d   matrix %.2f MB, x %.2f MB\n",
              argv[1], m, n, K.nnz(), (K.nnz() * 12.0 + m * 4.0) / 1e6, n * 8.0 / 1e6);

  std::vector<double> x(sz(n)), y(sz(m), 0.0), xt(sz(n), 0.0);
  for (Int j = 0; j < n; ++j) x[sz(j)] = 1.0 + 0.001 * (j % 97);

  int reps = 1;
  for (;;) {
    double el = timed(reps, [&] { spmv_rows(K, x.data(), y.data(), 0, m); });
    if (el * reps > 0.20) break;
    reps *= 2;
    if (reps > (1 << 22)) break;
  }

  // What this machine can actually stream, for the bandwidth question. Sized
  // well past any cache so it is DRAM and not the SLC.
  {
    const std::size_t N = 32u << 20;  // 256 MB per array
    std::vector<double> a(N, 1.0), b(N, 2.0), c(N, 0.0);
    std::printf("\n%-8s %12s\n", "threads", "triad GB/s");
    for (int t = 1; t <= maxt; ++t) {
      Pool pool(t);
      std::vector<std::size_t> cut(sz(t) + 1);
      for (int i = 0; i <= t; ++i) cut[sz(i)] = N * static_cast<std::size_t>(i) / static_cast<std::size_t>(t);
      std::function<void(int)> job = [&](int id) {
        for (std::size_t i = cut[sz(id)]; i < cut[sz(id) + 1]; ++i) c[i] = a[i] + 3.0 * b[i];
      };
      for (int i = 0; i < 3; ++i) pool.run(job);
      double el = 1e30;
      for (int r = 0; r < 5; ++r) el = std::min(el, timed(3, [&] { pool.run(job); }));
      std::printf("%-8d %12.1f\n", t, 3.0 * static_cast<double>(N) * 8.0 / el / 1e9);
    }
  }

  // Interleaved trials, minimum per thread count. The box is shared and a
  // mean would carry whatever else was running; the minimum is the closest
  // thing to an uncontended run that a shared machine can report, and taking
  // the thread counts in a different order each trial stops one burst of
  // interference landing entirely on one row.
  const int trials = 5;
  std::vector<double> best_sk(sz(maxt) + 1, 1e30), best_dk(sz(maxt) + 1, 1e30);
  std::vector<double> best_skt(sz(maxt) + 1, 1e30), best_dkt(sz(maxt) + 1, 1e30);
  for (int trial = 0; trial < trials; ++trial) {
    for (int step = 0; step < maxt; ++step) {
      const int t = 1 + ((step + trial) % maxt);
      Pool pool(t);
      const std::vector<Int> cs = split_by_nnz(K, t), cst = split_by_nnz(Kt, t);
      // Dynamic: a fixed number of blocks, independent of the thread count,
      // taken by whoever is free. Each block writes its own rows, so the answer
      // does not depend on who took what.
      const int blocks = std::max(t * 8, 16);
      const std::vector<Int> bk = split_by_nnz(K, blocks), bkt = split_by_nnz(Kt, blocks);
      std::atomic<int> next{0};

      std::function<void(int)> stat_k = [&](int id) { spmv_rows(K, x.data(), y.data(), cs[sz(id)], cs[sz(id) + 1]); };
      std::function<void(int)> stat_kt = [&](int id) { spmv_rows(Kt, y.data(), xt.data(), cst[sz(id)], cst[sz(id) + 1]); };
      std::function<void(int)> dyn_k = [&](int) {
        for (;;) { int b = next.fetch_add(1, std::memory_order_relaxed);
                   if (b >= blocks) return; spmv_rows(K, x.data(), y.data(), bk[sz(b)], bk[sz(b) + 1]); } };
      std::function<void(int)> dyn_kt = [&](int) {
        for (;;) { int b = next.fetch_add(1, std::memory_order_relaxed);
                   if (b >= blocks) return; spmv_rows(Kt, y.data(), xt.data(), bkt[sz(b)], bkt[sz(b) + 1]); } };
      auto run_dyn = [&](std::function<void(int)>& j) { next.store(0, std::memory_order_relaxed); pool.run(j); };

      for (int i = 0; i < 30; ++i) { pool.run(stat_k); run_dyn(dyn_k); }

      best_sk[sz(t)]  = std::min(best_sk[sz(t)],  timed(reps, [&] { pool.run(stat_k); }));
      best_dk[sz(t)]  = std::min(best_dk[sz(t)],  timed(reps, [&] { run_dyn(dyn_k); }));
      best_skt[sz(t)] = std::min(best_skt[sz(t)], timed(reps, [&] { pool.run(stat_kt); }));
      best_dkt[sz(t)] = std::min(best_dkt[sz(t)], timed(reps, [&] { run_dyn(dyn_kt); }));
    }
  }
  std::printf("\n# best of %d interleaved trials\n", trials);
  std::printf("%-8s %10s %10s %8s %8s %10s %10s %8s %8s\n", "threads",
              "Kx stat", "Kx dyn", "sp stat", "sp dyn", "Kty stat", "Kty dyn", "sp stat", "sp dyn");
  for (int t = 1; t <= maxt; ++t) {
    std::printf("%-8d %10.1f %10.1f %8.2f %8.2f %10.1f %10.1f %8.2f %8.2f\n", t,
                1e6 * best_sk[sz(t)], 1e6 * best_dk[sz(t)],
                best_sk[1] / best_sk[sz(t)], best_sk[1] / best_dk[sz(t)],
                1e6 * best_skt[sz(t)], 1e6 * best_dkt[sz(t)],
                best_skt[1] / best_skt[sz(t)], best_skt[1] / best_dkt[sz(t)]);
  }

  std::printf("\n%-8s %12s\n", "threads", "barrier us");
  for (int t = 1; t <= maxt; ++t) {
    Pool pool(t);
    std::function<void(int)> nop = [](int) {};
    for (int i = 0; i < 2000; ++i) pool.run(nop);
    double el = 1e30;
    for (int r = 0; r < 5; ++r) el = std::min(el, timed(20000, [&] { pool.run(nop); }));
    std::printf("%-8d %12.3f\n", t, 1e6 * el);
  }
  return 0;
}
