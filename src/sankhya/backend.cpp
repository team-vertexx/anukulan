#include "sankhya/backend.hpp"

#include <algorithm>
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

  double* allocate(Int n) const override {
    return n > 0 ? new double[sz(n)]() : nullptr;
  }
  void deallocate(double* p) const override { delete[] p; }
  void upload(const double* host, double* target, Int n) const override {
    std::copy(host, host + sz(n), target);
  }
  void download(const double* source, double* host, Int n) const override {
    std::copy(source, source + sz(n), host);
  }
  void fill(double* target, Int n, double value) const override {
    std::fill(target, target + sz(n), value);
  }
  void copy(const double* source, double* target, Int n) const override {
    std::copy(source, source + sz(n), target);
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

  // The three lines that used to be three separate operations are written here
  // in the order they used to run in - step, then reflection, then blend - and
  // both are skipped when they are identity. That is not stylistic: it keeps
  // this bit-identical to the sequence it replaces, so the iteration counts on
  // the CPU do not move and the fusion can be judged on the GPU alone.
  void primal_update(Int n, double tau, double gamma, double halpern,
                     const double* x, const double* c, const double* kt_y,
                     const double* lower, const double* upper,
                     const double* anchor, double* x_out, double* dx,
                     double* x_bar) const override {
    for (Int j = 0; j < n; ++j) {
      const std::size_t sj = sz(j);
      const double x_j = x[sj];
      double v = x_j - tau * (c[sj] - kt_y[sj]);
      if (v < lower[sj]) v = lower[sj];
      if (v > upper[sj]) v = upper[sj];
      const double d = v - x_j;
      dx[sj] = d;
      x_bar[sj] = v + d;
      double next = v;
      if (gamma != 0.0) next += gamma * d;
      if (anchor != nullptr) next = halpern * next + (1.0 - halpern) * anchor[sj];
      x_out[sj] = next;
    }
  }

  void dual_update(Int m, Int num_equalities, double sigma, double gamma,
                   double halpern, const double* y, const double* q,
                   const double* k_x_bar, const double* k_x, const double* anchor_y,
                   const double* anchor_kx, double* y_out, double* dy, double* k_out,
                   double* k_dx) const override {
    for (Int i = 0; i < m; ++i) {
      const std::size_t si = sz(i);
      const double y_i = y[si];
      const double bar = k_x_bar[si];
      const double kx = k_x[si];
      double v = y_i + sigma * (q[si] - bar);
      if (i >= num_equalities && v < 0.0) v = 0.0;
      const double d = v - y_i;
      dy[si] = d;
      const double kd = 0.5 * (bar - kx);
      k_dx[si] = kd;

      double next_y = v;
      double next_k = 0.5 * (bar + kx);
      if (gamma != 0.0) {
        next_y += gamma * d;
        next_k += gamma * kd;
      }
      if (anchor_y != nullptr) {
        next_y = halpern * next_y + (1.0 - halpern) * anchor_y[si];
        next_k = halpern * next_k + (1.0 - halpern) * anchor_kx[si];
      }
      y_out[si] = next_y;
      k_out[si] = next_k;
    }
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

void LinAlgBackend::step_size_terms(Int n, Int m, const double* dx,
                                    const double* dy, const double* k_dx,
                                    double omega, double* interaction,
                                    double* movement) const {
  // The absolute value belongs here, not at the call site: the step is safe
  // while eta * |dy' K dx| stays under half the weighted movement, and a
  // backend that returned the signed value would let a negative interaction
  // grow eta without bound.
  *interaction = std::fabs(dot(dy, k_dx, m));
  *movement = weighted_norm_squared(n, m, dx, dy, omega);
}

void LinAlgBackend::convergence_terms(Int n, Int m, const ConvergenceProblem& lp,
                                      const double* x, const double* k_x,
                                      const double* y, const double* kt_y,
                                      ConvergenceTerms* out) const {
  // The reference. Term for term and in the same order as evaluate_residual,
  // because that is what makes the two agree exactly rather than nearly - and
  // an unscaled residual that nearly agrees is a solver that stops in a
  // different place on the two backends.
  ConvergenceTerms t;

  for (Int i = 0; i < m; ++i) {
    const std::size_t si = sz(i);
    const double scale = lp.row_scale[si];
    const double row = scale != 0.0 ? k_x[si] / scale : k_x[si];
    // Equality rows are violated in either direction; inequality rows only
    // when the activity falls below the right-hand side.
    const double violation =
        (i < lp.num_equalities) ? (row - lp.q[si]) : std::fmin(row - lp.q[si], 0.0);
    t.primal_sq += violation * violation;
    t.primal_inf = std::fmax(t.primal_inf, std::fabs(violation));
    t.q_dot_y += lp.q[si] * (scale * y[si]);
  }

  for (Int j = 0; j < n; ++j) {
    const std::size_t sj = sz(j);
    const double scale = lp.col_scale[sj];
    const double product = scale != 0.0 ? kt_y[sj] / scale : kt_y[sj];
    const double lambda = lp.c[sj] - product;

    const bool has_lo = lp.lower[sj] > -kInf;
    const bool has_hi = lp.upper[sj] < kInf;
    double leftover = 0.0;
    if (has_lo && has_hi) {
      leftover = 0.0;  // a boxed variable absorbs any sign
    } else if (has_lo) {
      leftover = std::fmin(lambda, 0.0);
    } else if (has_hi) {
      leftover = std::fmax(lambda, 0.0);
    } else {
      leftover = lambda;  // free variables need a zero reduced cost
    }
    t.dual_sq += leftover * leftover;
    t.dual_inf = std::fmax(t.dual_inf, std::fabs(leftover));

    // Minimising lambda_j x_j over the variable's own interval picks whichever
    // end the sign favours. The branch is on the sign and not on the bound
    // because an absorbed value of zero against an infinite bound is the one
    // combination that would produce a NaN.
    const double absorbed = lambda - leftover;
    if (absorbed > 0.0) {
      t.bound_term += absorbed * lp.lower[sj];
    } else if (absorbed < 0.0) {
      t.bound_term += absorbed * lp.upper[sj];
    }

    t.c_dot_x += lp.c[sj] * (scale * x[sj]);
  }

  *out = t;
}

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
