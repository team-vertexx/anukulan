#pragma once

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

  // omega * ||dx||^2 + ||dy||^2 / omega, the weighted norm PDLP works in.
  virtual double weighted_norm_squared(Int n, Int m, const double* dx,
                                       const double* dy, double omega) const = 0;
};

// Plain scalar C++. The reference against which any other backend is checked.
const LinAlgBackend& cpu_backend();

}  // namespace sankhya
