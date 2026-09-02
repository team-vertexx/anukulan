#pragma once

#include <algorithm>
#include <string>
#include <vector>

#include "sankhya/sparse.hpp"

namespace sankhya {

// Every numerical operation the first-order solver performs, behind one narrow
// interface.
//
// The point is the CUDA port. The algorithm - adaptive step size, restarts,
// primal weight - is intricate and was hard to get right; the arithmetic under
// it is five kernels. Separating them means the algorithm is written once,
// debugged once on a machine with no GPU, and the port becomes a matter of
// reimplementing this interface rather than rewriting the solver.
//
// The operations are deliberately coarse. A backend that offered only axpy and
// dot would force the loop to make a dozen round trips per iteration, which is
// exactly the wrong shape for a GPU. Instead each fused step below is one kernel
// launch: read the vectors once, write them once.
class LinAlgBackend {
 public:
  virtual ~LinAlgBackend() = default;

  virtual std::string name() const = 0;

  // A backend that keeps its own copy of a matrix - on a device, say - needs to
  // know when one is about to be used many times, and when it can be dropped.
  // The CPU backend ignores both. Without this the CUDA backend would have to
  // guess whether a matrix it has seen before is still the same one, and branch
  // and bound builds a fresh problem at every node, so guessing would be wrong.
  virtual void prepare(const SparseMatrix& a) const { (void)a; }
  virtual void release() const {}

  // Working memory, owned by the backend.
  //
  // This is the difference between a GPU port that pays and one that does not.
  // If the operations take host pointers, every one of them has to copy its
  // arguments across and its results back, and on a first-order method that is
  // roughly ten transfers per iteration - which is exactly what the first
  // version of the CUDA backend did, and why it measured slower than the CPU it
  // was supposed to accelerate.
  //
  // With the vectors allocated here, they stay where the kernels are. The
  // solver moves data across only when it genuinely needs to look at it, which
  // is at the convergence check, every fortieth iteration.
  //
  // The CPU backend allocates ordinary host memory, so the same solver code
  // runs unchanged on both.
  virtual double* allocate(Int n) const = 0;
  virtual void deallocate(double* p) const = 0;
  virtual void upload(const double* host, double* target, Int n) const = 0;
  virtual void download(const double* source, double* host, Int n) const = 0;
  virtual void fill(double* target, Int n, double value) const = 0;
  virtual void copy(const double* source, double* target, Int n) const = 0;

  // y = A * x
  virtual void multiply(const SparseMatrix& a, const double* x, double* y) const = 0;
  // x = A^T * y, given A^T directly so both directions are a plain row sweep.
  virtual void multiply_transpose(const SparseMatrix& at, const double* y,
                                  double* x) const = 0;

  virtual double dot(const double* a, const double* b, Int n) const = 0;
  virtual double two_norm(const double* a, Int n) const = 0;
  virtual double inf_norm(const double* a, Int n) const = 0;

  // The whole primal side of one reflected Halpern PDHG step, in one operation:
  //
  //   x_pdhg = clamp(x - tau (c - kt_y), lower, upper)      the PDHG step
  //   dx     = x_pdhg - x
  //   x_bar  = 2 x_pdhg - x                                 the extrapolated point
  //   x_out  = halpern (x_pdhg + gamma dx) + (1 - halpern) anchor
  //
  // The last line is the two accelerations the solver actually runs, folded in.
  // `gamma` is the reflection - gamma = 0 is plain PDHG, gamma = 1 is the
  // Peaceman-Rachford point 2 x_pdhg - x. `anchor` is the Halpern anchor and
  // `halpern` its weight (k+1)/(k+2); pass `anchor == nullptr` for no blend,
  // which is what the first step of an epoch does.
  //
  // These used to be three operations - the step, an accumulate for the
  // reflection, a blend for the anchor - and on the GPU that is three kernels
  // reading and writing the same vector. cuPDLPx does the whole primal side in
  // one kernel, and that is why: the arithmetic is trivial, the traffic is not.
  //
  // `x_out` may alias `x`; every input is read once before anything is written.
  virtual void primal_update(Int n, double tau, double gamma, double halpern,
                             const double* x, const double* c, const double* kt_y,
                             const double* lower, const double* upper,
                             const double* anchor, double* x_out, double* dx,
                             double* x_bar) const = 0;

  // The dual side of the same step, and K applied to the point it produces:
  //
  //   y_pdhg = project(y + sigma (q - k_x_bar))   free on the first
  //                                               num_equalities entries,
  //                                               clamped at zero after
  //   dy     = y_pdhg - y
  //   y_out  = halpern (y_pdhg + gamma dy) + (1 - halpern) anchor_y
  //
  //   k_dx   = (k_x_bar - k_x) / 2                          = K dx
  //   k_out  = halpern ((k_x_bar + k_x)/2 + gamma k_dx) + (1 - halpern) anchor_kx
  //
  // The k_x half is where a whole sparse product goes. K x_bar = 2 K x_pdhg -
  // K x, so both K x_pdhg and K dx fall out of two vectors the step already
  // holds; and since K is linear, the reflection and the Halpern blend can be
  // applied to K z as well as to z rather than being recomputed. Recomputing it
  // instead costs one sparse product per iteration, which on this set was the
  // whole difference between Halpern winning and losing.
  //
  // `y_out` may alias `y` and `k_out` may alias `k_x`.
  virtual void dual_update(Int m, Int num_equalities, double sigma, double gamma,
                           double halpern, const double* y, const double* q,
                           const double* k_x_bar, const double* k_x,
                           const double* anchor_y, const double* anchor_kx,
                           double* y_out, double* dy, double* k_out,
                           double* k_dx) const = 0;

  // sum += weight * v, the step-size weighted running average.
  virtual void accumulate(Int n, double weight, const double* v,
                          double* sum) const = 0;
  // out = sum / weight
  virtual void scale_into(Int n, double weight, const double* sum,
                          double* out) const = 0;

  // omega * ||dx||^2 + ||dy||^2 / omega, the weighted norm PDLP works in.
  virtual double weighted_norm_squared(Int n, Int m, const double* dx,
                                       const double* dy, double omega) const = 0;

  // The three inner products the adaptive step size needs every iteration:
  //
  //     interaction = |dy' K dx|
  //     movement    = omega ||dx||^2 + ||dy||^2 / omega
  //
  // They are asked for together rather than one at a time because of what that
  // costs on a GPU. Each reduction ends in a device-to-host copy, and a copy is
  // a synchronisation: the pipeline drains, and the cost is latency rather than
  // bytes. Three of them per iteration is three stalls per iteration, which on
  // qap15 - 24,720 iterations - is around a second of doing nothing, roughly a
  // sixth of the whole solve. Fused, it is one stall.
  //
  // The default implementation below is the obvious one, and is what the CPU
  // uses; a backend only overrides this if fusing actually buys it something.
  virtual void step_size_terms(Int n, Int m, const double* dx, const double* dy,
                               const double* k_dx, double omega,
                               double* interaction, double* movement) const;

  // The unscaled problem, as the convergence test needs to see it. The solver
  // iterates on a scaled copy and stops on the original, so every quantity
  // below is the original one and the two scale vectors are what maps between
  // them: x = D2 x~, y = D1 y~, K x = (K~ x~) / D1, K' y = (K~' y~) / D2.
  //
  // All of it is uploaded once and left alone, exactly like c, q and the
  // bounds the iteration itself reads.
  struct ConvergenceProblem {
    const double* c = nullptr;          // n
    const double* lower = nullptr;      // n
    const double* upper = nullptr;      // n
    const double* q = nullptr;          // m
    const double* row_scale = nullptr;  // m, the D1 diagonal
    const double* col_scale = nullptr;  // n, the D2 diagonal
    Int num_equalities = 0;
  };

  // Everything the termination test is built from, as seven numbers.
  struct ConvergenceTerms {
    double primal_sq = 0.0;    // sum of squared row violations
    double primal_inf = 0.0;   // largest absolute row violation
    double q_dot_y = 0.0;      // q' y
    double dual_sq = 0.0;      // sum of squared leftover reduced costs
    double dual_inf = 0.0;     // largest absolute leftover reduced cost
    double bound_term = 0.0;   // what the bounds contribute to the dual objective
    double c_dot_x = 0.0;      // c' x
  };

  // The convergence test, computed where the iterate already is.
  //
  // This is the other half of keeping the working set on the device. The loop
  // had already stopped moving vectors across the bus, but the check still
  // pulled five of them back - x, y, dy, K x and K' y - and then walked them on
  // the host, ten times, for seven numbers. On graph40-40 that is most of the
  // 79% of solve time that was outside the kernels.
  //
  // Written as one operation rather than seven because of what the seven cost
  // separately: each reduction ends in a device-to-host copy, and a copy is a
  // synchronisation. Fused, the whole check is two kernels and 56 bytes.
  //
  // `x`, `y`, `k_x` and `kt_y` are the scaled iterate and its two products -
  // that is, exactly what the iteration is already holding. Nothing here
  // recomputes a product: `k_x` is K~ x~ at the end of every step, and `kt_y`
  // is the transpose product the next iteration was going to make anyway.
  //
  // The default implementation is the reference, and is what the CPU uses. It
  // is the same arithmetic in the same order as evaluate_residual, so the two
  // paths agree to the bit and the CPU numbers do not move.
  virtual void convergence_terms(Int n, Int m, const ConvergenceProblem& lp,
                                 const double* x, const double* k_x,
                                 const double* y, const double* kt_y,
                                 ConvergenceTerms* out) const;

  // Per-kernel timing, for finding out which one owns the run.
  //
  // This exists rather than a note saying "profile it with nsys" because nsys
  // is not on the free GPU boxes and installing it there is its own afternoon.
  // A backend already knows what it is launching and when, so it can time
  // itself, and the answer arrives with the run instead of after a second one.
  //
  // Timing serialises the launches it measures, so a profiled run is slower
  // than a real one and its wall clock is not a benchmark. The proportions are
  // what it is for. A backend with nothing to say returns an empty report.
  virtual void set_profiling(bool on) const { (void)on; }
  virtual std::string profile_report() const { return {}; }
};

// Plain scalar C++. The reference against which any other backend is checked.
const LinAlgBackend& cpu_backend();

#ifdef SANKHYA_WITH_CUDA
// Hand-written CUDA kernels. Throws if no device is present.
const LinAlgBackend& cuda_backend();
#endif

// Returns the CUDA backend when this build has one and a device is present, and
// the CPU backend otherwise. Never throws: a machine without a GPU simply gets
// the reference implementation, which is what makes the same binary usable on a
// laptop and on a GPU box.
const LinAlgBackend& default_backend();

// Holds a backend allocation for as long as it is needed. The solver keeps a
// dozen of these and never sees a raw pointer's lifetime.
class BackendVector {
 public:
  BackendVector() = default;
  BackendVector(const LinAlgBackend& backend, Int n)
      : backend_(&backend), size_(n), data_(backend.allocate(n)) {}
  ~BackendVector() {
    if (backend_ && data_) backend_->deallocate(data_);
  }
  BackendVector(const BackendVector&) = delete;
  BackendVector& operator=(const BackendVector&) = delete;
  BackendVector(BackendVector&& other) noexcept { swap(other); }
  BackendVector& operator=(BackendVector&& other) noexcept {
    swap(other);
    return *this;
  }

  double* data() { return data_; }
  const double* data() const { return data_; }
  Int size() const { return size_; }

  void upload(const std::vector<double>& host) {
    backend_->upload(host.data(), data_, size_);
  }
  void download(std::vector<double>* host) const {
    host->resize(sz(size_));
    backend_->download(data_, host->data(), size_);
  }

 private:
  void swap(BackendVector& other) {
    std::swap(backend_, other.backend_);
    std::swap(size_, other.size_);
    std::swap(data_, other.data_);
  }
  const LinAlgBackend* backend_ = nullptr;
  Int size_ = 0;
  double* data_ = nullptr;
};

}  // namespace sankhya
