#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#include "sankhya/sparse.hpp"

namespace sankhya {

// A small pool of worker threads, created once and reused for the life of a
// solve.
//
// Why this exists rather than OpenMP. Three reasons, in the order they decided
// it:
//
//   1. The reductions have to be bit-identical, and OpenMP's reduction clause
//      makes no such promise - the combining order is the runtime's business
//      and may vary with the thread count. Every determinism claim in
//      backend.hpp rests on controlling the partition ourselves.
//   2. Apple Clang ships without libomp. Adding OpenMP means a Homebrew
//      dependency on every machine that builds this, which breaks the one
//      thing the build currently promises: clone it and run cmake.
//   3. The project's premise is that the stack is its own. A thread pool is
//      ninety lines; the sparse LU is three hundred.
//
// Two things in here are measurements rather than taste, both on the M4 this
// was written on (4 performance cores, 6 efficiency cores):
//
//   Blocks are taken dynamically, not dealt out one per thread. On an even
//   static split every barrier waits for whichever chunk landed on an
//   efficiency core, and that chunk takes about three times as long. A
//   streaming triad measures 84 GB/s on three threads and 36 GB/s on ten under
//   a static split - adding six cores made it less than half as fast. On K*x
//   for datt256_lp, static peaks at 2.64x and dynamic reaches 3.15x.
//
//   Workers spin briefly and then yield. Pure spinning measured a ten-thread
//   barrier at 6,161 microseconds, against 1.2 at nine: with every core
//   occupied the scheduler takes a spinner off its core and everyone waits out
//   its quantum. With the yield it is 0.14 to 0.73 microseconds from two
//   threads to nine, and still 16.8 at ten - which is why default_thread_count
//   leaves a core alone.
class ThreadPool {
 public:
  explicit ThreadPool(int threads);
  ~ThreadPool();

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  int size() const { return threads_; }

  // Runs body(b) for every b in [0, blocks), spread over the pool, and returns
  // once all of them have finished. Blocks are handed out on demand, so which
  // thread runs which block is not fixed - every caller below therefore has to
  // make sure the answer does not depend on that.
  void run_blocks(int blocks, const std::function<void(int)>& body);

 private:
  void worker(int id);

  int threads_;
  std::vector<std::thread> workers_;
  const std::function<void(int)>* body_ = nullptr;
  int blocks_ = 0;
  std::atomic<int> next_block_{0};
  std::atomic<int> finished_{0};
  std::atomic<unsigned> generation_{0};
  std::atomic<bool> stopping_{false};
};

// Threads the solver should use when it is not told a number. One below the
// core count: the last core is what the barrier measurement above says not to
// take, and anything else on the box has to run somewhere.
int default_thread_count();

// Row boundaries splitting `matrix` into `blocks` pieces of roughly equal
// nonzero count. Rows are wildly uneven - splitting by row count leaves some
// blocks with almost no work - and the boundaries are rows, so each block
// writes its own entries of the output and no two blocks touch the same one.
std::vector<Int> split_rows_by_nonzeros(const SparseMatrix& matrix, int blocks);

// The same for a plain elementwise loop over n entries.
std::vector<Int> split_range(Int n, int blocks);

}  // namespace sankhya
