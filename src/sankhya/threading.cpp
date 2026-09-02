#include "sankhya/threading.hpp"

#include <algorithm>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

namespace sankhya {
namespace {

// How long a worker spins before it gives up and sleeps.
//
// It has to cover the gap between one parallel loop and the next, and that gap
// is the part of the iteration that runs on the calling thread - the inner
// products, which are deliberately serial. Too short and the workers sleep
// through every serial section and are woken again for every kernel, which
// costs a wake-up per iteration.
//
// Swept on maros-r7, a tight first-order loop, against gt2 through the MILP
// path, where the pool is built and then barely used because the node LPs are
// small enough to run serially:
//
//     spin limit   maros-r7 t=2   maros-r7 t=4   gt2 milp t=6
//      1,000          1.200 s        1.465 s        1.888 s
//      4,000          1.180 s        1.407 s        1.922 s
//     20,000          1.115 s        1.075 s        1.922 s
//    100,000          1.072 s        0.941 s        1.930 s
//    400,000          1.071 s        0.942 s        1.900 s
//
// The tight loop wants a long spin - 100,000 is 36% faster than 1,000 at four
// threads - and the idle pool does not care, because the spin happens once and
// then the worker sleeps until it is woken. That asymmetry is why the number
// can be this large without the cost the gt2 column was there to catch. Past
// 100,000 nothing more is bought.
constexpr int kSpinLimit = 100000;

}  // namespace

ThreadPool::ThreadPool(int threads) : threads_(std::max(1, threads)) {
  workers_.reserve(sz(static_cast<Int>(threads_ - 1)));
  for (int i = 1; i < threads_; ++i) {
    workers_.emplace_back([this, i] { worker(i); });
  }
}

ThreadPool::~ThreadPool() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stopping_.store(true, std::memory_order_relaxed);
    generation_.fetch_add(1, std::memory_order_release);
  }
  wake_.notify_all();
  for (std::thread& t : workers_) t.join();
}

void ThreadPool::worker(int) {
  unsigned seen = 0;
  for (;;) {
    // Spin first, because inside a solve the next call is microseconds away and
    // a condition variable round trip is not.
    bool ready = false;
    for (int spins = 0; spins < kSpinLimit; ++spins) {
      if (stopping_.load(std::memory_order_relaxed)) return;
      if (generation_.load(std::memory_order_acquire) != seen) {
        ready = true;
        break;
      }
    }
    // Then sleep, because outside a solve the next call may never come, and a
    // worker spinning on an idle pool is a core taken away from whatever is
    // actually running.
    if (!ready) {
      std::unique_lock<std::mutex> lock(mutex_);
      sleepers_.fetch_add(1, std::memory_order_release);
      wake_.wait(lock, [this, seen] {
        return stopping_.load(std::memory_order_relaxed) ||
               generation_.load(std::memory_order_acquire) != seen;
      });
      sleepers_.fetch_sub(1, std::memory_order_release);
    }
    if (stopping_.load(std::memory_order_relaxed)) return;
    seen = generation_.load(std::memory_order_acquire);

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

  {
    // Under the lock so a worker cannot decide to sleep between reading the
    // generation and waiting on it.
    std::lock_guard<std::mutex> lock(mutex_);
    body_ = &body;
    blocks_ = blocks;
    next_block_.store(0, std::memory_order_relaxed);
    finished_.store(0, std::memory_order_relaxed);
    generation_.fetch_add(1, std::memory_order_release);
  }
  // Only pay for the wake-up when somebody is actually asleep. Inside a solve
  // the workers are still in their spin phase and this is skipped.
  if (sleepers_.load(std::memory_order_acquire) > 0) wake_.notify_all();

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

  // Not hardware_concurrency, and not one less than it either - both are wrong
  // here, and the measurement in docs/RESULTS.md 10.2 is what says so. On this
  // machine the geometric mean speedup peaks at 1.32x on four threads and is a
  // *loss* at nine and ten (0.87x and 0.81x). A default that took every core
  // would ship a flag that makes the solver slower when it is asked for.
  //
  // The reason is in 10.3: the loop is memory-bound and one core already moves
  // three quarters of what the bus can deliver, so the extra threads are
  // queueing for bandwidth rather than working. That ceiling is reached at
  // about the point the performance cores run out, which is also where the
  // efficiency cores start dragging every barrier out to their own speed.
  //
  // So: the performance core count where the system will say what it is, and
  // half the logical cores otherwise, which is the closest guess available on a
  // machine that does not distinguish them. This is a starting point, not a
  // tuned value - the right number is a property of the machine's bandwidth per
  // core, and anyone who cares should run bench/thread_scaling.py on theirs.
#if defined(__APPLE__)
  int performance = 0;
  std::size_t size = sizeof(performance);
  if (sysctlbyname("hw.perflevel0.logicalcpu", &performance, &size, nullptr, 0) == 0 &&
      performance > 0) {
    return std::min(performance, static_cast<int>(hw));
  }
#endif
  return std::max(1, static_cast<int>(hw) / 2);
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
