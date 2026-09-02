// Where does threading an elementwise kernel start to pay?
//
// threaded_backend.cpp has two size thresholds below which it runs on the
// calling thread. Both were guesses, and the per-kernel profile in
// docs/RESULTS.md 10.6 showed the guess was wrong in the direction that costs:
// advance_kx and dual_step, which run over a vector of eleven thousand entries,
// got *slower* the more threads they were given.
//
// This measures the crossover directly, against the serial backend, for the two
// shapes that matter - the fused primal step, which is the widest elementwise
// kernel in the loop, and a sparse product.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "sankhya/backend.hpp"
#include "sankhya/sparse.hpp"

using namespace sankhya;
using Clock = std::chrono::steady_clock;

template <typename F>
static double best_of(int trials, int reps, F&& f) {
  double best = 1e30;
  for (int t = 0; t < trials; ++t) {
    const auto t0 = Clock::now();
    for (int r = 0; r < reps; ++r) f();
    const double el = std::chrono::duration<double>(Clock::now() - t0).count() / reps;
    if (el < best) best = el;
  }
  return best;
}

int main(int argc, char** argv) {
  const int maxt = argc > 1 ? std::atoi(argv[1]) : 8;

  std::printf("# fused primal step, microseconds per call, best of 20\n");
  std::printf("%-10s %10s", "n", "serial");
  for (int t = 2; t <= maxt; ++t) std::printf(" %9d", t);
  std::printf("   %10s\n", "best sp");

  for (const Int n : {1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288}) {
    std::vector<double> x(sz(n), 1.5), c(sz(n), 0.25), kt(sz(n), 0.75);
    std::vector<double> lo(sz(n), -1.0), hi(sz(n), 2.0);
    std::vector<double> xn(sz(n)), dx(sz(n)), xb(sz(n));
    auto call = [&](const LinAlgBackend& b) {
      b.primal_step(n, 0.37, x.data(), c.data(), kt.data(), lo.data(), hi.data(),
                    xn.data(), dx.data(), xb.data());
    };
    const int reps = n < 32768 ? 2000 : 200;
    const double serial = best_of(20, reps, [&] { call(cpu_backend()); });
    std::printf("%-10d %10.2f", n, 1e6 * serial);
    double best = serial;
    for (int t = 2; t <= maxt; ++t) {
      const LinAlgBackend& b = threaded_cpu_backend(t);
      for (int w = 0; w < 200; ++w) call(b);  // warm the pool
      const double el = best_of(20, reps, [&] { call(b); });
      std::printf(" %9.2f", 1e6 * el);
      if (el < best) best = el;
    }
    std::printf("   %10.2f\n", serial / best);
  }

  std::printf("\n# sparse product K x, microseconds per call, best of 20\n");
  std::printf("%-10s %10s %10s", "nnz", "rows", "serial");
  for (int t = 2; t <= maxt; ++t) std::printf(" %9d", t);
  std::printf("   %10s\n", "best sp");

  for (const Int rows : {512, 1024, 4096, 16384, 65536, 262144}) {
    const Int per_row = 8;
    std::vector<Triplet> entries;
    entries.reserve(sz(rows * per_row));
    for (Int r = 0; r < rows; ++r) {
      for (Int k = 0; k < per_row; ++k) {
        entries.push_back(Triplet{r, (r * 7 + k * 13) % rows, 1.0 + 0.1 * k});
      }
    }
    const SparseMatrix a = SparseMatrix::from_triplets(rows, rows, std::move(entries));
    std::vector<double> x(sz(rows), 1.25), y(sz(rows), 0.0);
    auto call = [&](const LinAlgBackend& b) { b.multiply(a, x.data(), y.data()); };
    const int reps = a.nnz() < 100000 ? 2000 : 200;
    const double serial = best_of(20, reps, [&] { call(cpu_backend()); });
    std::printf("%-10d %10d %10.2f", a.nnz(), rows, 1e6 * serial);
    double best = serial;
    for (int t = 2; t <= maxt; ++t) {
      const LinAlgBackend& b = threaded_cpu_backend(t);
      for (int w = 0; w < 200; ++w) call(b);
      const double el = best_of(20, reps, [&] { call(b); });
      std::printf(" %9.2f", 1e6 * el);
      if (el < best) best = el;
    }
    std::printf("   %10.2f\n", serial / best);
  }
  return 0;
}
