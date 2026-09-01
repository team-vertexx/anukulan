// What this machine can move, and how much of that one core can move alone.
//
// This is the number the whole of RESULTS 10.3 rests on, so it is measured on
// its own rather than as a preamble to something else: many short trials, the
// best kept per thread count, because on a shared box the best observed run is
// the closest thing to an uncontended one and a mean is mostly the neighbours.
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

using Clock = std::chrono::steady_clock;

int main(int argc, char** argv) {
  const int maxt = argc > 1 ? std::atoi(argv[1]) : 10;
  const int trials = argc > 2 ? std::atoi(argv[2]) : 15;
  const std::size_t N = 48u << 20;  // 384 MB per array, far past any cache

  std::vector<double> a(N, 1.0), b(N, 2.0), c(N, 0.0);
  std::vector<double> best(static_cast<std::size_t>(maxt) + 1, 0.0);

  for (int trial = 0; trial < trials; ++trial) {
    for (int step = 0; step < maxt; ++step) {
      // Rotate the order so one burst of interference cannot land on one row.
      const int t = 1 + ((step + trial) % maxt);
      std::vector<std::thread> workers;
      auto chunk = [&](int id) {
        const std::size_t lo = N * static_cast<std::size_t>(id) / static_cast<std::size_t>(t);
        const std::size_t hi = N * static_cast<std::size_t>(id + 1) / static_cast<std::size_t>(t);
        for (std::size_t i = lo; i < hi; ++i) c[i] = a[i] + 3.0 * b[i];
      };
      const auto t0 = Clock::now();
      for (int i = 1; i < t; ++i) workers.emplace_back(chunk, i);
      chunk(0);
      for (std::thread& w : workers) w.join();
      const double el = std::chrono::duration<double>(Clock::now() - t0).count();
      const double gbs = 3.0 * static_cast<double>(N) * 8.0 / el / 1e9;
      if (gbs > best[static_cast<std::size_t>(t)]) best[static_cast<std::size_t>(t)] = gbs;
    }
  }

  std::printf("%-8s %10s %10s\n", "threads", "GB/s", "vs 1");
  for (int t = 1; t <= maxt; ++t) {
    std::printf("%-8d %10.1f %10.2f\n", t, best[static_cast<std::size_t>(t)],
                best[static_cast<std::size_t>(t)] / best[1]);
  }
  return 0;
}
