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

  // The PDHG primal step, fused:
  //   x_next = clamp(x - tau * (c - kt_y), lower, upper)
  //   dx     = x_next - x
  //   x_bar  = 2 * x_next - x
  virtual void primal_step(Int n, double tau, const double* x, const double* c,
                           const double* kt_y, const double* lower,
                           const double* upper, double* x_next, double* dx,
                           double* x_bar) const = 0;

  // The PDHG dual step, fused:
  //   y_next = project(y + sigma * (q - k_x_bar))     free on the first
  //                                                   num_equalities entries,
  //                                                   clamped at zero after
  //   dy     = y_next - y
  //   k_dx   = (k_x_bar - k_x) / 2
  //
  // That last line is where the extra matrix-vector product goes. Since
  // K x_bar = 2 K x_next - K x, the product K(x_next - x) that the adaptive step
  // size rule needs falls out of quantities already computed.
  virtual void dual_step(Int m, Int num_equalities, double sigma, const double* y,
                         const double* q, const double* k_x_bar, const double* k_x,
                         double* y_next, double* dy, double* k_dx) const = 0;

  //   k_x = (k_x_bar + k_x) / 2, which is K applied to the accepted iterate.
  virtual void advance_kx(Int m, const double* k_x_bar, double* k_x) const = 0;

  // sum += weight * v, the step-size weighted running average.
  virtual void accumulate(Int n, double weight, const double* v,
                          double* sum) const = 0;
  // out = sum / weight
  virtual void scale_into(Int n, double weight, const double* sum,
                          double* out) const = 0;

  // z <- a z + b anchor, in place. The Halpern iteration's whole update.
  virtual void blend(Int n, double a, double* z, double b,
                     const double* anchor) const = 0;

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
