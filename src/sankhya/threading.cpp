#include "sankhya/threading.hpp"

#include <algorithm>

namespace sankhya {
namespace {

// How long a worker spins before yielding. Long enough to catch the next block
// of a loop that is already running, short enough that a descheduled peer does
// not cost a whole quantum.
constexpr int kSpinLimit = 4000;

}  // namespace

ThreadPool::ThreadPool(int threads) : threads_(std::max(1, threads)) {
  workers_.reserve(sz(static_cast<Int>(threads_ - 1)));
  for (int i = 1; i < threads_; ++i) {
    workers_.emplace_back([this, i] { worker(i); });
  }
}

ThreadPool::~ThreadPool() {
  stopping_.store(true, std::memory_order_relaxed);
  generation_.fetch_add(1, std::memory_order_release);
  for (std::thread& t : workers_) t.join();
}

void ThreadPool::worker(int) {
  unsigned seen = 0;
  for (;;) {
    int spins = 0;
    while (generation_.load(std::memory_order_acquire) == seen) {
      if (stopping_.load(std::memory_order_relaxed)) return;
      if (++spins > kSpinLimit) {
        std::this_thread::yield();
        spins = kSpinLimit - 1000;
      }
    }
    seen = generation_.load(std::memory_order_acquire);
    if (stopping_.load(std::memory_order_relaxed)) return;

    for (;;) {
      const int b = next_block_.fetch_add(1, std::memory_order_relaxed);
      if (b >= blocks_) break;
      (*body_)(b);
    }
    finished_.fetch_add(1, std::memory_order_release);
  }
}

void ThreadPool::run_blocks(int blocks, const std::function<void(int)>& body) {
  if (blocks <= 0) return;
  if (threads_ == 1) {
    for (int b = 0; b < blocks; ++b) body(b);
    return;
  }

  body_ = &body;
  blocks_ = blocks;
  next_block_.store(0, std::memory_order_relaxed);
  finished_.store(0, std::memory_order_relaxed);
  generation_.fetch_add(1, std::memory_order_release);

  // The calling thread is a worker too, which is what keeps a one-block loop
  // from costing a wake-up.
  for (;;) {
    const int b = next_block_.fetch_add(1, std::memory_order_relaxed);
    if (b >= blocks_) break;
    body(b);
  }

  int spins = 0;
  while (finished_.load(std::memory_order_acquire) != threads_ - 1) {
    if (++spins > kSpinLimit) {
      std::this_thread::yield();
      spins = kSpinLimit - 1000;
    }
  }
  body_ = nullptr;
}

int default_thread_count() {
  const unsigned hw = std::thread::hardware_concurrency();
  if (hw <= 2) return 1;
  return static_cast<int>(hw) - 1;
}

std::vector<Int> split_rows_by_nonzeros(const SparseMatrix& matrix, int blocks) {
  blocks = std::max(1, blocks);
  std::vector<Int> cut(sz(static_cast<Int>(blocks) + 1), 0);
  const Int rows = matrix.rows();
  const Int nnz = matrix.nnz();
  Int at = 0;
  for (int b = 1; b < blocks; ++b) {
    const Int target =
        static_cast<Int>((static_cast<long long>(nnz) * b) / blocks);
    while (at < rows && matrix.row_end(at) < target) ++at;
    cut[sz(static_cast<Int>(b))] = at;
  }
  cut[sz(static_cast<Int>(blocks))] = rows;
  // Monotone even when a single row holds more nonzeros than a whole share.
  for (int b = 1; b <= blocks; ++b) {
    if (cut[sz(static_cast<Int>(b))] < cut[sz(static_cast<Int>(b - 1))]) {
      cut[sz(static_cast<Int>(b))] = cut[sz(static_cast<Int>(b - 1))];
    }
  }
  return cut;
}

std::vector<Int> split_range(Int n, int blocks) {
  blocks = std::max(1, blocks);
  std::vector<Int> cut(sz(static_cast<Int>(blocks) + 1), 0);
  for (int b = 0; b <= blocks; ++b) {
    cut[sz(static_cast<Int>(b))] =
        static_cast<Int>((static_cast<long long>(n) * b) / blocks);
  }
  return cut;
}

}  // namespace sankhya
