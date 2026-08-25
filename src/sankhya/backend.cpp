#include "sankhya/backend.hpp"

#include <cmath>

namespace sankhya {
namespace {

class CpuBackend final : public LinAlgBackend {
 public:
  std::string name() const override { return "cpu"; }

  void multiply(const SparseMatrix& a, const double* x, double* y) const override {
    a.multiply(x, y);
  }

  void multiply_transpose(const SparseMatrix& at, const double* y,
                          double* x) const override {
    at.multiply(y, x);
  }

  double dot(const double* a, const double* b, Int n) const override {
    double sum = 0.0;
    for (Int i = 0; i < n; ++i) sum += a[sz(i)] * b[sz(i)];
    return sum;
  }

  double two_norm(const double* a, Int n) const override {
    return std::sqrt(dot(a, a, n));
  }

  double inf_norm(const double* a, Int n) const override {
    double m = 0.0;
    for (Int i = 0; i < n; ++i) m = std::fmax(m, std::fabs(a[sz(i)]));
    return m;
  }

  void primal_step(Int n, double tau, const double* x, const double* c,
                   const double* kt_y, const double* lower, const double* upper,
                   double* x_next, double* dx, double* x_bar) const override {
    for (Int j = 0; j < n; ++j) {
      const std::size_t sj = sz(j);
      double v = x[sj] - tau * (c[sj] - kt_y[sj]);
      if (v < lower[sj]) v = lower[sj];
      if (v > upper[sj]) v = upper[sj];
      x_next[sj] = v;
      const double d = v - x[sj];
      dx[sj] = d;
      x_bar[sj] = v + d;
    }
  }

  void dual_step(Int m, Int num_equalities, double sigma, const double* y,
                 const double* q, const double* k_x_bar, const double* k_x,
                 double* y_next, double* dy, double* k_dx) const override {
    for (Int i = 0; i < m; ++i) {
      const std::size_t si = sz(i);
      double v = y[si] + sigma * (q[si] - k_x_bar[si]);
      if (i >= num_equalities && v < 0.0) v = 0.0;
      y_next[si] = v;
      dy[si] = v - y[si];
      k_dx[si] = 0.5 * (k_x_bar[si] - k_x[si]);
    }
  }

  void advance_kx(Int m, const double* k_x_bar, double* k_x) const override {
    for (Int i = 0; i < m; ++i) k_x[sz(i)] = 0.5 * (k_x_bar[sz(i)] + k_x[sz(i)]);
  }

  void accumulate(Int n, double weight, const double* v,
                  double* sum) const override {
    for (Int i = 0; i < n; ++i) sum[sz(i)] += weight * v[sz(i)];
  }

  void scale_into(Int n, double weight, const double* sum,
                  double* out) const override {
    for (Int i = 0; i < n; ++i) out[sz(i)] = sum[sz(i)] / weight;
  }

  double weighted_norm_squared(Int n, Int m, const double* dx, const double* dy,
                               double omega) const override {
    return omega * dot(dx, dx, n) + dot(dy, dy, m) / omega;
  }
};

}  // namespace

const LinAlgBackend& cpu_backend() {
  static const CpuBackend backend;
  return backend;
}

const LinAlgBackend& default_backend() {
#ifdef SANKHYA_WITH_CUDA
  try {
    return cuda_backend();
  } catch (...) {
    // Built with CUDA but running where there is no device. Fall back rather
    // than fail: the same binary has to work on the laptop the code is written
    // on and on the box it is benchmarked on.
    return cpu_backend();
  }
#else
  return cpu_backend();
#endif
}

}  // namespace sankhya
