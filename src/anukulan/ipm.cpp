#include "anukulan/ipm.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <mutex>
#include <numeric>
#include <thread>
#include <unordered_map>

#include "anukulan/model.hpp"
#include "anukulan/presolve.hpp"

namespace anukulan {

std::string to_string(IpmStatus status) {
  switch (status) {
    case IpmStatus::kOptimal: return "optimal";
    case IpmStatus::kPrimalInfeasible: return "primal infeasible";
    case IpmStatus::kDualInfeasible: return "dual infeasible";
    case IpmStatus::kIterationLimit: return "iteration limit";
    case IpmStatus::kTimeLimit: return "time limit";
    case IpmStatus::kInterrupted: return "interrupted";
    case IpmStatus::kNumericalError: return "numerical error";
  }
  return "unknown";
}

namespace {

double inf_norm(const std::vector<double>& v) {
  double r = 0.0;
  for (const double x : v) r = std::fmax(r, std::fabs(x));
  return r;
}

double dot(const std::vector<double>& a, const std::vector<double>& b) {
  double s = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
  return s;
}

// What kind of bounds an internal variable has, after shifting and reflecting:
// v >= 0, 0 <= v <= ub, or none.
enum class Kind : unsigned char { kLower, kBox, kFree };

// Where an internal row came from.
enum class RowKind : unsigned char {
  kEquality,        // an equality row of the standard form
  kInequality,      // one >= row, with a slack w >= 0
  kRange,           // a >= row and its negation, one row with 0 <= w <= hi - lo
  kRangeEquality,   // the same pair with hi == lo, so no slack at all
};

// The problem the method iterates on:
//
//   min  c'v + constant
//   s.t. A v = b
//        v_j >= 0 (kLower), 0 <= v_j <= ub_j (kBox), free (kFree)
//
// v holds the kept structural columns first, then one slack per inequality or
// range row, whose column in A is -e_row.
struct Problem {
  Int m = 0;
  Int n = 0;
  Int n_struct = 0;
  SparseMatrix a;   // m x n
  SparseMatrix at;  // n x m, the column view
  std::vector<double> b;
  std::vector<double> c;
  std::vector<double> ub;
  std::vector<Kind> kind;
  double constant = 0.0;

  // Back to the scaled standard form: x_j = shift_j + sign_j * v_k with
  // k = internal_column[j], or x_j = shift_j when the column was fixed.
  std::vector<Int> internal_column;
  std::vector<double> shift;
  std::vector<double> sign;

  std::vector<RowKind> row_kind;
  std::vector<Int> first_row;
  std::vector<Int> second_row;
  std::vector<double> partner_factor;

  Int ranges = 0;
  Int fixed = 0;
  Int free = 0;
};

// Whether standard row `other` is a positive multiple of the negation of row
// `base`, and by what factor. A two-sided model row arrives as exactly such a
// pair; after scaling the two rows share a row norm and so a scale factor, but
// the test does not rely on that.
bool negated_pair(const StandardLp& s, Int base, Int other, double* factor) {
  const Int b0 = s.k.row_begin(base);
  const Int b1 = s.k.row_end(base);
  const Int o0 = s.k.row_begin(other);
  const Int o1 = s.k.row_end(other);
  if (b1 - b0 != o1 - o0 || b1 == b0) return false;
  const double alpha = -s.k.value()[sz(o0)] / s.k.value()[sz(b0)];
  if (!(alpha > 0.0) || !std::isfinite(alpha)) return false;
  for (Int e = 0; e < b1 - b0; ++e) {
    if (s.k.index()[sz(b0 + e)] != s.k.index()[sz(o0 + e)]) return false;
    const double vb = s.k.value()[sz(b0 + e)];
    const double vo = s.k.value()[sz(o0 + e)];
    if (std::fabs(vo + alpha * vb) > 1e-12 * std::fabs(vo)) return false;
  }
  *factor = alpha;
  return true;
}

void build_problem(const StandardLp& s, Problem* p) {
  const Int n_std = s.num_cols();
  const Int m_std = s.num_rows();

  p->internal_column.assign(sz(n_std), -1);
  p->shift.assign(sz(n_std), 0.0);
  p->sign.assign(sz(n_std), 1.0);
  p->constant = 0.0;
  for (Int j = 0; j < n_std; ++j) {
    const double lo = s.lower[sz(j)];
    const double hi = s.upper[sz(j)];
    const bool has_lo = lo > -kInf;
    const bool has_hi = hi < kInf;
    if (has_lo && has_hi && lo == hi) {
      p->shift[sz(j)] = lo;
      p->constant += s.c[sz(j)] * lo;
      ++p->fixed;
      continue;
    }
    Kind kind = Kind::kFree;
    double upper = kInf;
    if (has_lo && has_hi) {
      kind = Kind::kBox;
      p->shift[sz(j)] = lo;
      upper = hi - lo;
    } else if (has_lo) {
      kind = Kind::kLower;
      p->shift[sz(j)] = lo;
    } else if (has_hi) {
      // Only an upper bound: reflect, v = hi - x >= 0.
      kind = Kind::kLower;
      p->shift[sz(j)] = hi;
      p->sign[sz(j)] = -1.0;
    } else {
      ++p->free;
    }
    p->internal_column[sz(j)] = static_cast<Int>(p->kind.size());
    p->kind.push_back(kind);
    p->ub.push_back(upper);
    p->c.push_back(p->sign[sz(j)] * s.c[sz(j)]);
    p->constant += s.c[sz(j)] * p->shift[sz(j)];
  }
  p->n_struct = static_cast<Int>(p->kind.size());

  // Pair up the two halves of each two-sided row.
  std::vector<Int> partner(sz(m_std), -1);
  std::vector<double> factor(sz(m_std), 0.0);
  if (static_cast<Int>(s.row_origin.size()) == m_std) {
    std::unordered_map<Int, Int> seen;
    for (Int i = s.num_equalities; i < m_std; ++i) {
      const Int key = s.row_origin[sz(i)].model_row;
      const auto found = seen.find(key);
      if (found == seen.end()) {
        seen.emplace(key, i);
        continue;
      }
      const Int base = found->second;
      double alpha = 0.0;
      if (partner[sz(base)] < 0 && negated_pair(s, base, i, &alpha)) {
        partner[sz(base)] = i;
        partner[sz(i)] = base;
        factor[sz(base)] = alpha;
      }
    }
  }

  std::vector<Triplet> entries;
  entries.reserve(sz(s.k.nnz()) + sz(m_std));
  Int slacks = 0;
  auto add_row = [&](Int base, RowKind kind, double slack_upper) {
    const Int r = static_cast<Int>(p->row_kind.size());
    double rhs = s.q[sz(base)];
    for (Int e = s.k.row_begin(base); e < s.k.row_end(base); ++e) {
      const Int j = s.k.index()[sz(e)];
      const double value = s.k.value()[sz(e)];
      rhs -= value * p->shift[sz(j)];
      const Int k = p->internal_column[sz(j)];
      if (k >= 0) entries.push_back({r, k, p->sign[sz(j)] * value});
    }
    if (kind == RowKind::kInequality || kind == RowKind::kRange) {
      entries.push_back({r, p->n_struct + slacks, -1.0});
      ++slacks;
      p->kind.push_back(kind == RowKind::kRange ? Kind::kBox : Kind::kLower);
      p->ub.push_back(slack_upper);
      p->c.push_back(0.0);
    }
    p->b.push_back(rhs);
    p->row_kind.push_back(kind);
    p->first_row.push_back(base);
    p->second_row.push_back(kind == RowKind::kRange || kind == RowKind::kRangeEquality
                                ? partner[sz(base)]
                                : -1);
    p->partner_factor.push_back(kind == RowKind::kRange || kind == RowKind::kRangeEquality
                                    ? factor[sz(base)]
                                    : 0.0);
  };
  for (Int i = 0; i < s.num_equalities; ++i) add_row(i, RowKind::kEquality, 0.0);
  for (Int i = s.num_equalities; i < m_std; ++i) {
    const Int other = partner[sz(i)];
    if (other >= 0 && other < i) continue;  // the second half of a pair
    if (other < 0) {
      add_row(i, RowKind::kInequality, kInf);
      continue;
    }
    // K_i x >= lo and -alpha K_i x >= q_other, so K_i x <= hi = -q_other / alpha.
    const double lo = s.q[sz(i)];
    const double hi = -s.q[sz(other)] / factor[sz(i)];
    const double width = hi - lo;
    if (width > 0.0) {
      add_row(i, RowKind::kRange, width);
      ++p->ranges;
    } else if (width >= -1e-12 * (1.0 + std::fabs(lo))) {
      add_row(i, RowKind::kRangeEquality, 0.0);
      ++p->ranges;
    } else {
      // An empty range. Leave the two rows as they are; the method will find
      // no feasible point and say so.
      partner[sz(other)] = -1;
      partner[sz(i)] = -1;
      add_row(i, RowKind::kInequality, kInf);
    }
  }
  // Second halves whose pair was undone above still need their own row.
  for (Int i = s.num_equalities; i < m_std; ++i) {
    const Int other = partner[sz(i)];
    if (other < 0 && factor[sz(i)] == 0.0) {
      bool present = false;
      for (const Int r : p->first_row) present = present || r == i;
      (void)present;
    }
  }
  p->m = static_cast<Int>(p->row_kind.size());
  p->n = p->n_struct + slacks;
  p->a = SparseMatrix::from_triplets(p->m, p->n, std::move(entries));
  p->at = p->a.transpose();
}

// The normal equations M = A Theta A' + delta I and everything needed to solve
// with them: the sparse part factorised, the dense columns put back through
// Sherman-Morrison-Woodbury, and conjugate gradients on the true M on top.
class NormalEquations {
 public:
  struct Report {
    Int iterations = 0;
    double relative_residual = 0.0;
  };

  void setup(const Problem& p, const IpmOptions& options) {
    p_ = &p;
    const Int m = p.m;
    dense_.assign(sz(p.n), 0);
    dense_list_.clear();
    if (m >= options.dense_column_rows && options.max_dense_columns > 0) {
      const double limit = std::fmax(static_cast<double>(options.dense_column_minimum),
                                     options.dense_column_fraction * static_cast<double>(m));
      std::vector<std::pair<Int, Int>> candidates;
      for (Int j = 0; j < p.n_struct; ++j) {
        const Int count = p.at.row_end(j) - p.at.row_begin(j);
        if (static_cast<double>(count) > limit) candidates.push_back({-count, j});
      }
      std::sort(candidates.begin(), candidates.end());
      for (std::size_t q = 0; q < candidates.size() &&
                              static_cast<Int>(q) < options.max_dense_columns;
           ++q) {
        dense_[sz(candidates[q].second)] = 1;
        dense_list_.push_back(candidates[q].second);
      }
    }

    // Lower triangle of the sparse part, diagonal always present.
    start_.assign(sz(m) + 1, 0);
    index_.clear();
    diagonal_.assign(sz(m), 0);
    std::vector<Int> mark(sz(m), -1);
    std::vector<Int> list;
    for (Int r = 0; r < m; ++r) {
      list.clear();
      mark[sz(r)] = r;
      list.push_back(r);
      for (Int e = p.a.row_begin(r); e < p.a.row_end(r); ++e) {
        const Int j = p.a.index()[sz(e)];
        if (dense_[sz(j)]) continue;
        for (Int f = p.at.row_begin(j); f < p.at.row_end(j); ++f) {
          const Int r2 = p.at.index()[sz(f)];
          if (r2 > r) break;
          if (mark[sz(r2)] != r) {
            mark[sz(r2)] = r;
            list.push_back(r2);
          }
        }
      }
      std::sort(list.begin(), list.end());
      for (const Int r2 : list) {
        if (r2 == r) diagonal_[sz(r)] = static_cast<Int>(index_.size());
        index_.push_back(r2);
      }
      start_[sz(r) + 1] = static_cast<Int>(index_.size());
    }
    values_.assign(index_.size(), 0.0);

    std::vector<Triplet> pattern;
    pattern.reserve(index_.size());
    for (Int r = 0; r < m; ++r) {
      for (Int q = start_[sz(r)]; q < start_[sz(r) + 1]; ++q)
        pattern.push_back({r, index_[sz(q)], 1.0});
    }
    const SparseMatrix lower = SparseMatrix::from_triplets(m, m, std::move(pattern));
    std::string ignored;
    chol_.analyse(lower, options.cholesky, &ignored);

    work_m_.assign(sz(m), 0.0);
    dense_diagonal_.assign(sz(m), 0.0);
    theta_.assign(sz(p.n), 0.0);
    max_cg_ = dense_list_.empty() ? 6 : 50 + 2 * static_cast<Int>(dense_list_.size());
  }

  // Assemble and factorise for this Theta. Returns the number of pivots the
  // factorisation dropped.
  Int factorize(const std::vector<double>& theta, double delta) {
    const Problem& p = *p_;
    const Int m = p.m;
    theta_ = theta;
    delta_ = delta;
    std::vector<double>& w = work_m_;
    for (Int r = 0; r < m; ++r) {
      for (Int e = p.a.row_begin(r); e < p.a.row_end(r); ++e) {
        const Int j = p.a.index()[sz(e)];
        if (dense_[sz(j)]) continue;
        const double t = p.a.value()[sz(e)] * theta[sz(j)];
        if (t == 0.0) continue;
        for (Int f = p.at.row_begin(j); f < p.at.row_end(j); ++f) {
          const Int r2 = p.at.index()[sz(f)];
          if (r2 > r) break;
          w[sz(r2)] += t * p.at.value()[sz(f)];
        }
      }
      for (Int q = start_[sz(r)]; q < start_[sz(r) + 1]; ++q) {
        values_[sz(q)] = w[sz(index_[sz(q)])];
        w[sz(index_[sz(q)])] = 0.0;
      }
      values_[sz(diagonal_[sz(r)])] += delta;
    }

    // A row the dense columns hold up almost alone has next to nothing on its
    // diagonal in the sparse part, and Woodbury built on that loses every
    // digit. So the preconditioner factorises M_s + B instead, B being the
    // dense columns' own diagonal on those rows; conjugate gradients on the
    // true M take B's effect back out, in about as many steps as B has rows.
    supported_rows_ = 0;
    if (!dense_list_.empty()) {
      std::fill(dense_diagonal_.begin(), dense_diagonal_.end(), 0.0);
      for (const Int j : dense_list_) {
        for (Int f = p.at.row_begin(j); f < p.at.row_end(j); ++f) {
          const double a = p.at.value()[sz(f)];
          dense_diagonal_[sz(p.at.index()[sz(f)])] += a * a * theta[sz(j)];
        }
      }
      for (Int r = 0; r < m; ++r) {
        double& d = values_[sz(diagonal_[sz(r)])];
        if (d < 1e-2 * dense_diagonal_[sz(r)]) {
          d += dense_diagonal_[sz(r)];
          ++supported_rows_;
        }
      }
    }
    chol_.factorize(values_);
    const Int dropped = chol_.dropped_pivots();

    woodbury_ = false;
    const std::size_t k = dense_list_.size();
    if (k > 0) {
      u_.assign(sz(m) * k, 0.0);
      w_.assign(sz(m) * k, 0.0);
      for (std::size_t q = 0; q < k; ++q) {
        const Int j = dense_list_[q];
        const double root = std::sqrt(theta[sz(j)]);
        double* column = u_.data() + q * sz(m);
        for (Int f = p.at.row_begin(j); f < p.at.row_end(j); ++f)
          column[p.at.index()[sz(f)]] = p.at.value()[sz(f)] * root;
        std::copy(column, column + m, w_.data() + q * sz(m));
        chol_.solve(w_.data() + q * sz(m));
      }
      // S = I + U' P^-1 U, then its Cholesky factor in place (lower).
      schur_.assign(k * k, 0.0);
      for (std::size_t a = 0; a < k; ++a) {
        for (std::size_t b2 = 0; b2 <= a; ++b2) {
          double sum = (a == b2) ? 1.0 : 0.0;
          const double* ua = u_.data() + a * sz(m);
          const double* wb = w_.data() + b2 * sz(m);
          for (Int r = 0; r < m; ++r) sum += ua[r] * wb[r];
          schur_[a * k + b2] = sum;
        }
      }
      woodbury_ = true;
      for (std::size_t j = 0; j < k && woodbury_; ++j) {
        double d = schur_[j * k + j];
        for (std::size_t q = 0; q < j; ++q) d -= schur_[j * k + q] * schur_[j * k + q];
        if (!(d > 0.0) || !std::isfinite(d)) {
          woodbury_ = false;
          break;
        }
        d = std::sqrt(d);
        schur_[j * k + j] = d;
        for (std::size_t i = j + 1; i < k; ++i) {
          double s = schur_[i * k + j];
          for (std::size_t q = 0; q < j; ++q) s -= schur_[i * k + q] * schur_[j * k + q];
          schur_[i * k + j] = s / d;
        }
      }
    }
    return dropped;
  }

  // M dy = rhs by preconditioned conjugate gradients. Without dense columns the
  // preconditioner is the factorisation of M itself and the first step is
  // already the answer; the later ones are iterative refinement in all but
  // name.
  Report solve(const std::vector<double>& rhs, std::vector<double>* dy) {
    const Int m = p_->m;
    Report report;
    dy->assign(sz(m), 0.0);
    const double scale = inf_norm(rhs);
    if (scale == 0.0 || m == 0) return report;
    r_ = rhs;
    precondition(r_, &z_);
    p_dir_ = z_;
    double rz = dot(r_, z_);
    best_ = *dy;
    double best_residual = scale;
    for (Int it = 0; it < max_cg_; ++it) {
      multiply(p_dir_, &q_);
      const double pq = dot(p_dir_, q_);
      if (!(pq > 0.0) || !std::isfinite(pq)) break;
      const double alpha = rz / pq;
      for (Int i = 0; i < m; ++i) {
        (*dy)[sz(i)] += alpha * p_dir_[sz(i)];
        r_[sz(i)] -= alpha * q_[sz(i)];
      }
      ++report.iterations;
      const double residual = inf_norm(r_);
      if (residual < best_residual) {
        best_residual = residual;
        best_ = *dy;
      }
      if (residual <= 1e-13 * scale || !std::isfinite(residual)) break;
      precondition(r_, &z_);
      const double rz_next = dot(r_, z_);
      if (!(rz_next > 0.0)) break;
      const double beta = rz_next / rz;
      rz = rz_next;
      for (Int i = 0; i < m; ++i) p_dir_[sz(i)] = z_[sz(i)] + beta * p_dir_[sz(i)];
    }
    *dy = best_;
    report.relative_residual = best_residual / scale;
    return report;
  }

  const CholeskyFactor& factor() const { return chol_; }
  Int dense_columns() const { return static_cast<Int>(dense_list_.size()); }
  long long pattern_nonzeros() const { return static_cast<long long>(index_.size()); }
  Int supported_rows() const { return supported_rows_; }

 private:
  // The true matrix: A Theta A' + delta I, dense columns included, no shift.
  void multiply(const std::vector<double>& x, std::vector<double>* out) {
    const Problem& p = *p_;
    work_n_.assign(sz(p.n), 0.0);
    p.at.multiply(x.data(), work_n_.data());
    for (Int j = 0; j < p.n; ++j) work_n_[sz(j)] *= theta_[sz(j)];
    out->assign(sz(p.m), 0.0);
    p.a.multiply(work_n_.data(), out->data());
    for (Int i = 0; i < p.m; ++i) (*out)[sz(i)] += delta_ * x[sz(i)];
  }

  void precondition(const std::vector<double>& r, std::vector<double>* z) {
    *z = r;
    chol_.solve(z);
    if (!woodbury_) return;
    const Int m = p_->m;
    const std::size_t k = dense_list_.size();
    small_.assign(k, 0.0);
    for (std::size_t q = 0; q < k; ++q) {
      const double* u = u_.data() + q * sz(m);
      double s = 0.0;
      for (Int i = 0; i < m; ++i) s += u[i] * (*z)[sz(i)];
      small_[q] = s;
    }
    for (std::size_t i = 0; i < k; ++i) {
      double s = small_[i];
      for (std::size_t q = 0; q < i; ++q) s -= schur_[i * k + q] * small_[q];
      small_[i] = s / schur_[i * k + i];
    }
    for (std::size_t i = k; i-- > 0;) {
      double s = small_[i];
      for (std::size_t q = i + 1; q < k; ++q) s -= schur_[q * k + i] * small_[q];
      small_[i] = s / schur_[i * k + i];
    }
    for (std::size_t q = 0; q < k; ++q) {
      const double* w = w_.data() + q * sz(m);
      const double h = small_[q];
      for (Int i = 0; i < m; ++i) (*z)[sz(i)] -= w[i] * h;
    }
  }

  const Problem* p_ = nullptr;
  std::vector<char> dense_;
  std::vector<Int> dense_list_;
  std::vector<Int> start_;
  std::vector<Int> index_;
  std::vector<Int> diagonal_;
  std::vector<double> values_;
  CholeskyFactor chol_;
  std::vector<double> theta_;
  double delta_ = 0.0;
  Int max_cg_ = 6;
  Int supported_rows_ = 0;
  bool woodbury_ = false;
  std::vector<double> u_;
  std::vector<double> w_;
  std::vector<double> schur_;
  std::vector<double> dense_diagonal_;
  std::vector<double> work_m_;
  std::vector<double> work_n_;
  std::vector<double> small_;
  std::vector<double> r_;
  std::vector<double> z_;
  std::vector<double> p_dir_;
  std::vector<double> q_;
  std::vector<double> best_;
};

struct Direction {
  std::vector<double> dv;
  std::vector<double> ds;
  std::vector<double> dy;
  std::vector<double> dz;
  std::vector<double> dt;

  void resize(Int n, Int m) {
    dv.assign(sz(n), 0.0);
    ds.assign(sz(n), 0.0);
    dy.assign(sz(m), 0.0);
    dz.assign(sz(n), 0.0);
    dt.assign(sz(n), 0.0);
  }
};

// Largest step in [0, inf) keeping x + alpha dx >= 0 on the masked entries.
double max_step(const std::vector<double>& x, const std::vector<double>& dx,
                const std::vector<char>& mask) {
  double alpha = std::numeric_limits<double>::infinity();
  for (std::size_t j = 0; j < x.size(); ++j) {
    if (!mask[j] || dx[j] >= 0.0) continue;
    alpha = std::fmin(alpha, -x[j] / dx[j]);
  }
  return alpha;
}

// A point of the internal problem back in the caller's standard form,
// unscaled. x is clipped onto its bounds and y onto the sign the standard
// form's dual needs, so what is measured is a point that could be handed on.
void to_standard(const Problem& p, const StandardLp& original, const Scaling& scaling,
                 const std::vector<double>& v, const std::vector<double>& y,
                 std::vector<double>* x_out, std::vector<double>* y_out) {
  const Int n_std = original.num_cols();
  const Int m_std = original.num_rows();
  x_out->assign(sz(n_std), 0.0);
  for (Int j = 0; j < n_std; ++j) {
    const Int k = p.internal_column[sz(j)];
    const double scaled = (k < 0) ? p.shift[sz(j)] : p.shift[sz(j)] + p.sign[sz(j)] * v[sz(k)];
    double value = scaling.col_scale[sz(j)] * scaled;
    value = std::fmax(value, original.lower[sz(j)]);
    value = std::fmin(value, original.upper[sz(j)]);
    (*x_out)[sz(j)] = value;
  }
  y_out->assign(sz(m_std), 0.0);
  for (Int r = 0; r < p.m; ++r) {
    const Int i = p.first_row[sz(r)];
    const double value = y[sz(r)];
    switch (p.row_kind[sz(r)]) {
      case RowKind::kEquality:
        (*y_out)[sz(i)] = value;
        break;
      case RowKind::kInequality:
        (*y_out)[sz(i)] = std::fmax(value, 0.0);
        break;
      case RowKind::kRange:
      case RowKind::kRangeEquality: {
        (*y_out)[sz(i)] = std::fmax(value, 0.0);
        const Int other = p.second_row[sz(r)];
        (*y_out)[sz(other)] = std::fmax(-value, 0.0) / p.partner_factor[sz(r)];
        break;
      }
    }
  }
  for (Int i = 0; i < m_std; ++i) (*y_out)[sz(i)] *= scaling.row_scale[sz(i)];
}

// Farkas: y >= 0 on the inequality rows with max over the box of y'K x below
// y'q proves no x satisfies the rows and the bounds. The parts of K'y that an
// infinite bound cannot absorb are the certificate's error, and it is accepted
// only when that error is tiny against the margin it proves.
bool primal_infeasibility_certificate(const StandardLp& lp, const std::vector<double>& ray,
                                      double tolerance) {
  const double size = inf_norm(ray);
  if (!(size > 0.0) || !std::isfinite(size)) return false;
  std::vector<double> lambda(sz(lp.num_cols()), 0.0);
  lp.kt.multiply(ray.data(), lambda.data());
  double support = 0.0;
  double error = 0.0;
  for (Int j = 0; j < lp.num_cols(); ++j) {
    const double l = lambda[sz(j)];
    if (l > 0.0) {
      if (lp.upper[sz(j)] < kInf) {
        support += l * lp.upper[sz(j)];
      } else {
        error = std::fmax(error, l);
      }
    } else if (l < 0.0) {
      if (lp.lower[sz(j)] > -kInf) {
        support += l * lp.lower[sz(j)];
      } else {
        error = std::fmax(error, -l);
      }
    }
  }
  const double margin = dot(lp.q, ray) - support;
  return margin > 0.0 && error <= tolerance * margin;
}

// An improving ray: K_eq d = 0, K_ineq d >= 0, d inside the recession cone of
// the bounds, c'd < 0. Proves the objective is unbounded below, provided the
// rows are feasible at all, which the method cannot see from here, so this is
// only reported as dual infeasibility.
bool dual_infeasibility_certificate(const StandardLp& lp, const std::vector<double>& d,
                                    double tolerance) {
  const double size = inf_norm(d);
  if (!(size > 0.0) || !std::isfinite(size)) return false;
  const double descent = -dot(lp.c, d);
  if (!(descent > 0.0)) return false;
  std::vector<double> kd(sz(lp.num_rows()), 0.0);
  lp.k.multiply(d.data(), kd.data());
  double error = 0.0;
  for (Int i = 0; i < lp.num_rows(); ++i) {
    const double v = kd[sz(i)];
    error = std::fmax(error, i < lp.num_equalities ? std::fabs(v) : std::fmax(-v, 0.0));
  }
  for (Int j = 0; j < lp.num_cols(); ++j) {
    if (lp.lower[sz(j)] > -kInf) error = std::fmax(error, -d[sz(j)]);
    if (lp.upper[sz(j)] < kInf) error = std::fmax(error, d[sz(j)]);
  }
  return error <= tolerance * descent;
}

}  // namespace

IpmResult solve_ipm(const StandardLp& lp, const IpmOptions& options) {
  const auto clock_start = std::chrono::steady_clock::now();
  auto elapsed = [&clock_start]() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - clock_start)
        .count();
  };
  auto stop_requested = [&options]() {
    return options.stop != nullptr && options.stop->load(std::memory_order_relaxed);
  };

  IpmResult result;
  std::string error;
  if (!lp.validate(&error)) {
    result.message = "the problem failed validation: " + error;
    return result;
  }

  StandardLp scaled = lp;
  Scaling scaling;
  scaling.row_scale.assign(sz(lp.num_rows()), 1.0);
  scaling.col_scale.assign(sz(lp.num_cols()), 1.0);
  if (options.scale) scaling = scale_lp(&scaled, options.scaling).scaling;

  Problem p;
  build_problem(scaled, &p);
  result.ranges_merged = p.ranges;
  result.fixed_columns = p.fixed;
  result.free_columns = p.free;

  const double analyse_start = elapsed();
  NormalEquations normal;
  normal.setup(p, options);
  result.analyse_seconds = elapsed() - analyse_start;
  result.normal_rows = p.m;
  result.normal_nonzeros = normal.pattern_nonzeros();
  result.factor_nonzeros = normal.factor().nonzeros();
  result.factor_flops = normal.factor().flops();
  result.supernodes = normal.factor().supernodes();
  result.dense_columns = normal.dense_columns();

  const Int m = p.m;
  const Int n = p.n;
  std::vector<char> has_lower(sz(n), 0);
  std::vector<char> has_upper(sz(n), 0);
  Int pairs = 0;
  for (Int j = 0; j < n; ++j) {
    has_lower[sz(j)] = p.kind[sz(j)] != Kind::kFree;
    has_upper[sz(j)] = p.kind[sz(j)] == Kind::kBox;
    pairs += has_lower[sz(j)] + has_upper[sz(j)];
  }

  std::vector<double> v(sz(n), 0.0);
  std::vector<double> s(sz(n), 0.0);
  std::vector<double> z(sz(n), 0.0);
  std::vector<double> t(sz(n), 0.0);
  std::vector<double> y(sz(m), 0.0);
  std::vector<double> theta(sz(n), 1.0);
  std::vector<double> work_n(sz(n), 0.0);
  std::vector<double> work_m(sz(m), 0.0);
  std::vector<double> rhs(sz(m), 0.0);
  const double delta = options.dual_regularization;
  double factor_time = 0.0;

  auto factorize = [&](double regularisation) {
    const double t0 = elapsed();
    result.dropped_pivots += normal.factorize(theta, regularisation);
    factor_time += elapsed() - t0;
  };
  auto solve_normal = [&](const std::vector<double>& right, std::vector<double>* out) {
    const NormalEquations::Report report = normal.solve(right, out);
    result.cg_iterations += report.iterations;
    return report;
  };

  // Mehrotra's starting point, generalised to boxes. The primal one is the
  // least-squares point of A v = b with each box pulled towards its middle,
  //   min |v|^2 + |s|^2  s.t.  A v = b,  v + s = ub,
  // whose solution is v = Theta A' lambda + ub/2 on boxes with Theta = 1/2
  // there and 1 elsewhere, and A Theta A' lambda = b - A ub/2. The dual one is
  // the least-squares y for A'y + z - t = c with the same weights. Both are
  // then shifted to be positive and balanced so neither side starts far more
  // central than the other. Mehrotra, "On the implementation of a primal-dual
  // interior point method", SIAM J. Optim. 2(4), 1992, section 7.
  {
    std::vector<double> half(sz(n), 0.0);
    for (Int j = 0; j < n; ++j) {
      theta[sz(j)] = has_upper[sz(j)] ? 0.5 : 1.0;
      if (has_upper[sz(j)]) half[sz(j)] = 0.5 * p.ub[sz(j)];
    }
    factorize(std::fmax(delta, 1e-8));
    p.a.multiply(half.data(), work_m.data());
    for (Int i = 0; i < m; ++i) rhs[sz(i)] = p.b[sz(i)] - work_m[sz(i)];
    std::vector<double> lambda;
    solve_normal(rhs, &lambda);
    p.at.multiply(lambda.data(), work_n.data());
    for (Int j = 0; j < n; ++j) {
      v[sz(j)] = theta[sz(j)] * work_n[sz(j)] + half[sz(j)];
      if (has_upper[sz(j)]) s[sz(j)] = p.ub[sz(j)] - v[sz(j)];
    }
    for (Int j = 0; j < n; ++j) work_n[sz(j)] = theta[sz(j)] * p.c[sz(j)];
    p.a.multiply(work_n.data(), rhs.data());
    solve_normal(rhs, &y);
    p.at.multiply(y.data(), work_n.data());
    for (Int j = 0; j < n; ++j) {
      const double r = p.c[sz(j)] - work_n[sz(j)];
      if (has_upper[sz(j)]) {
        z[sz(j)] = 0.5 * r;
        t[sz(j)] = -0.5 * r;
      } else if (has_lower[sz(j)]) {
        z[sz(j)] = r;
      }
    }
    double primal_min = kInf;
    double dual_min = kInf;
    for (Int j = 0; j < n; ++j) {
      if (has_lower[sz(j)]) {
        primal_min = std::fmin(primal_min, v[sz(j)]);
        dual_min = std::fmin(dual_min, z[sz(j)]);
      }
      if (has_upper[sz(j)]) {
        primal_min = std::fmin(primal_min, s[sz(j)]);
        dual_min = std::fmin(dual_min, t[sz(j)]);
      }
    }
    const double primal_shift = pairs > 0 ? std::fmax(-1.5 * primal_min, 0.0) : 0.0;
    const double dual_shift = pairs > 0 ? std::fmax(-1.5 * dual_min, 0.0) : 0.0;
    double product = 0.0;
    double primal_sum = 0.0;
    double dual_sum = 0.0;
    for (Int j = 0; j < n; ++j) {
      if (has_lower[sz(j)]) {
        v[sz(j)] += primal_shift;
        z[sz(j)] += dual_shift;
        product += v[sz(j)] * z[sz(j)];
        primal_sum += v[sz(j)];
        dual_sum += z[sz(j)];
      }
      if (has_upper[sz(j)]) {
        s[sz(j)] += primal_shift;
        t[sz(j)] += dual_shift;
        product += s[sz(j)] * t[sz(j)];
        primal_sum += s[sz(j)];
        dual_sum += t[sz(j)];
      }
    }
    const double primal_balance = dual_sum > 0.0 ? 0.5 * product / dual_sum : 0.0;
    const double dual_balance = primal_sum > 0.0 ? 0.5 * product / primal_sum : 0.0;
    for (Int j = 0; j < n; ++j) {
      if (has_lower[sz(j)]) {
        v[sz(j)] += primal_balance;
        z[sz(j)] += dual_balance;
        if (!(v[sz(j)] > 0.0)) v[sz(j)] = 1.0;
        if (!(z[sz(j)] > 0.0)) z[sz(j)] = 1.0;
      }
      if (has_upper[sz(j)]) {
        s[sz(j)] += primal_balance;
        t[sz(j)] += dual_balance;
        if (!(s[sz(j)] > 0.0)) s[sz(j)] = 1.0;
        if (!(t[sz(j)] > 0.0)) t[sz(j)] = 1.0;
      }
    }
  }

  std::vector<double> rp(sz(m), 0.0);
  std::vector<double> ru(sz(n), 0.0);
  std::vector<double> rd(sz(n), 0.0);
  std::vector<double> r1(sz(n), 0.0);
  std::vector<double> r2(sz(n), 0.0);
  std::vector<double> h(sz(n), 0.0);
  const std::vector<double> zero_m(sz(m), 0.0);
  const std::vector<double> zero_n(sz(n), 0.0);
  Direction affine;
  Direction direction;
  Direction correction;
  affine.resize(n, m);
  direction.resize(n, m);
  correction.resize(n, m);

  // One Newton system for the right-hand sides given, with the factorisation
  // already in place. Everything but dy is recovered by substitution.
  auto newton = [&](const std::vector<double>& primal, const std::vector<double>& upper,
                    const std::vector<double>& dual, const std::vector<double>& comp_lower,
                    const std::vector<double>& comp_upper, Direction* d) {
    for (Int j = 0; j < n; ++j) {
      double value = dual[sz(j)];
      if (has_lower[sz(j)]) value -= comp_lower[sz(j)] / v[sz(j)];
      if (has_upper[sz(j)])
        value += (comp_upper[sz(j)] - t[sz(j)] * upper[sz(j)]) / s[sz(j)];
      h[sz(j)] = value;
      work_n[sz(j)] = theta[sz(j)] * value;
    }
    p.a.multiply(work_n.data(), work_m.data());
    for (Int i = 0; i < m; ++i) rhs[sz(i)] = primal[sz(i)] + work_m[sz(i)];
    solve_normal(rhs, &d->dy);
    p.at.multiply(d->dy.data(), work_n.data());
    bool finite = true;
    for (Int j = 0; j < n; ++j) {
      const double dv = theta[sz(j)] * (work_n[sz(j)] - h[sz(j)]);
      d->dv[sz(j)] = dv;
      if (has_upper[sz(j)]) {
        d->ds[sz(j)] = upper[sz(j)] - dv;
        d->dt[sz(j)] = (comp_upper[sz(j)] - t[sz(j)] * d->ds[sz(j)]) / s[sz(j)];
      } else {
        d->ds[sz(j)] = 0.0;
        d->dt[sz(j)] = 0.0;
      }
      d->dz[sz(j)] = has_lower[sz(j)] ? (comp_lower[sz(j)] - z[sz(j)] * dv) / v[sz(j)] : 0.0;
      finite = finite && std::isfinite(dv) && std::isfinite(d->dz[sz(j)]) &&
               std::isfinite(d->dt[sz(j)]);
    }
    for (Int i = 0; i < m; ++i) finite = finite && std::isfinite(d->dy[sz(i)]);
    return finite;
  };
  auto primal_step = [&](const Direction& d) {
    return std::fmin(max_step(v, d.dv, has_lower), max_step(s, d.ds, has_upper));
  };
  auto dual_step = [&](const Direction& d) {
    return std::fmin(max_step(z, d.dz, has_lower), max_step(t, d.dt, has_upper));
  };
  auto complementarity_after = [&](const Direction& d, double ap, double ad) {
    double sum = 0.0;
    for (Int j = 0; j < n; ++j) {
      if (has_lower[sz(j)])
        sum += (v[sz(j)] + ap * d.dv[sz(j)]) * (z[sz(j)] + ad * d.dz[sz(j)]);
      if (has_upper[sz(j)])
        sum += (s[sz(j)] + ap * d.ds[sz(j)]) * (t[sz(j)] + ad * d.dt[sz(j)]);
    }
    return pairs > 0 ? sum / pairs : 0.0;
  };

  std::vector<double> x_std;
  std::vector<double> y_std;
  std::vector<double> previous_x;
  std::vector<double> previous_y;
  double best_measure = kInf;
  IpmStatus status = IpmStatus::kIterationLimit;
  Int infeasible_streak = 0;
  Int unbounded_streak = 0;
  Int stalled = 0;
  Int raises = 0;
  double regularisation = delta;
  double mu = 0.0;
  const double c_norm = 1.0 + inf_norm(lp.c);
  (void)c_norm;

  Int iteration = 0;
  for (;; ++iteration) {
    // Residuals of the internal problem.
    p.a.multiply(v.data(), work_m.data());
    for (Int i = 0; i < m; ++i) rp[sz(i)] = p.b[sz(i)] - work_m[sz(i)];
    p.at.multiply(y.data(), work_n.data());
    double product = 0.0;
    for (Int j = 0; j < n; ++j) {
      ru[sz(j)] = has_upper[sz(j)] ? p.ub[sz(j)] - v[sz(j)] - s[sz(j)] : 0.0;
      rd[sz(j)] = p.c[sz(j)] - work_n[sz(j)] - z[sz(j)] + t[sz(j)];
      if (has_lower[sz(j)]) product += v[sz(j)] * z[sz(j)];
      if (has_upper[sz(j)]) product += s[sz(j)] * t[sz(j)];
    }
    mu = pairs > 0 ? product / pairs : 0.0;

    // Judged on the caller's problem, not on this one.
    to_standard(p, lp, scaling, v, y, &x_std, &y_std);
    const PdhgResidual measured = evaluate_residual(lp, x_std, y_std);
    const double measure = measured.worst_relative(true);
    if (measure < best_measure || result.x.empty()) {
      if (measure < 0.5 * best_measure) stalled = 0;
      best_measure = std::fmin(best_measure, measure);
      result.x = x_std;
      result.y = y_std;
      result.residual = measured;
    } else {
      ++stalled;
    }
    result.final_mu = mu / (1.0 + std::fabs(measured.primal_objective));
    if (options.verbose) {
      std::printf(
          "  ipm %3d  pobj %+.10e  dobj %+.10e  primal %.2e  dual %.2e  gap %.2e  mu %.2e\n",
          iteration, lp.objective_scale * measured.primal_objective + lp.objective_offset,
          lp.objective_scale * measured.dual_objective + lp.objective_offset,
          std::fmax(measured.relative_primal, measured.relative_primal_inf),
          std::fmax(measured.relative_dual, measured.relative_dual_inf),
          measured.relative_gap, mu);
    }
    if (measured.converged(options.tolerance, 0.0, true, 0.0)) {
      status = IpmStatus::kOptimal;
      result.x = x_std;
      result.y = y_std;
      result.residual = measured;
      break;
    }
    // Cancellation and the time limit are checked here rather than at the top
    // of the loop, and that placement is load-bearing rather than
    // cosmetic: result.x and result.y are only ever assigned above, so a
    // caller stopped before that assignment ever ran would be handed the
    // empty vectors IpmResult starts with, and model_objective on an empty x
    // reads past the end of lp.c. Checking after guarantees at least the
    // starting point has been measured and recorded before either can end
    // the loop. The race between engines (`anukulan lp`) is what makes this
    // reachable: a losing IPM run can be cancelled on its very first
    // iteration, before it has ever factorised anything.
    if (stop_requested()) {
      status = IpmStatus::kInterrupted;
      break;
    }
    if (elapsed() > options.time_limit_seconds) {
      status = IpmStatus::kTimeLimit;
      break;
    }
    if (iteration >= options.max_iterations) {
      status = IpmStatus::kIterationLimit;
      break;
    }

    // Certificates. Checked on the iterate itself, which grows along the ray
    // when there is one, and on the change since the last iterate.
    if (options.detect_infeasibility && iteration >= 5) {
      const double tol = 1e-9;
      bool primal_ray = measured.relative_primal_inf > options.tolerance &&
                        primal_infeasibility_certificate(lp, y_std, tol);
      if (!primal_ray && !previous_y.empty() && measured.relative_primal_inf > options.tolerance) {
        std::vector<double> dy(y_std.size());
        for (std::size_t i = 0; i < dy.size(); ++i) dy[i] = y_std[i] - previous_y[i];
        for (Int i = lp.num_equalities; i < lp.num_rows(); ++i)
          dy[sz(i)] = std::fmax(dy[sz(i)], 0.0);
        primal_ray = primal_infeasibility_certificate(lp, dy, tol);
      }
      infeasible_streak = primal_ray ? infeasible_streak + 1 : 0;
      bool dual_ray = measured.relative_dual_inf > options.tolerance &&
                      dual_infeasibility_certificate(lp, x_std, tol);
      if (!dual_ray && !previous_x.empty() && measured.relative_dual_inf > options.tolerance) {
        std::vector<double> dx(x_std.size());
        for (std::size_t j = 0; j < dx.size(); ++j) dx[j] = x_std[j] - previous_x[j];
        dual_ray = dual_infeasibility_certificate(lp, dx, tol);
      }
      unbounded_streak = dual_ray ? unbounded_streak + 1 : 0;
      if (infeasible_streak >= 3) {
        status = IpmStatus::kPrimalInfeasible;
        result.message = "the dual iterates converge to a Farkas certificate";
        break;
      }
      if (unbounded_streak >= 3) {
        status = IpmStatus::kDualInfeasible;
        result.message = "the primal iterates converge to an improving ray";
        break;
      }
    }
    previous_x = x_std;
    previous_y = y_std;

    // Mehrotra has stopped making progress: the point is as good as this
    // arithmetic will make it.
    if (stalled >= 8 && mu < 1e-12 * (1.0 + std::fabs(measured.primal_objective))) {
      status = IpmStatus::kNumericalError;
      result.message = "stalled: no progress in eight iterations";
      break;
    }

    for (Int j = 0; j < n; ++j) {
      double diagonal = 0.0;
      if (has_lower[sz(j)]) diagonal += z[sz(j)] / v[sz(j)];
      if (has_upper[sz(j)]) diagonal += t[sz(j)] / s[sz(j)];
      diagonal += p.kind[sz(j)] == Kind::kFree ? options.free_regularization
                                               : options.primal_regularization;
      theta[sz(j)] = 1.0 / diagonal;
    }
    factorize(regularisation);
    if (stop_requested()) {
      status = IpmStatus::kInterrupted;
      break;
    }

    // Predictor: the affine scaling direction, complementarity driven to zero.
    for (Int j = 0; j < n; ++j) {
      r1[sz(j)] = has_lower[sz(j)] ? -v[sz(j)] * z[sz(j)] : 0.0;
      r2[sz(j)] = has_upper[sz(j)] ? -s[sz(j)] * t[sz(j)] : 0.0;
    }
    bool finite = newton(rp, ru, rd, r1, r2, &affine);
    if (!finite) {
      // Raise the regularisation and try once more before giving up.
      if (raises < 3) {
        ++raises;
        ++result.regularisation_raises;
        regularisation = std::fmax(regularisation * 1e3, 1e-8);
        continue;
      }
      status = IpmStatus::kNumericalError;
      result.message = "the Newton direction is not finite";
      break;
    }
    const double ap_aff = std::fmin(1.0, primal_step(affine));
    const double ad_aff = std::fmin(1.0, dual_step(affine));
    const double mu_aff = complementarity_after(affine, ap_aff, ad_aff);
    double sigma = mu > 0.0 ? std::pow(mu_aff / mu, 3.0) : 0.0;
    sigma = std::fmin(1.0, std::fmax(0.0, sigma));

    // Corrector: centre towards sigma mu, with Mehrotra's second-order term.
    const double target = sigma * mu;
    for (Int j = 0; j < n; ++j) {
      r1[sz(j)] = has_lower[sz(j)]
                      ? target - v[sz(j)] * z[sz(j)] - affine.dv[sz(j)] * affine.dz[sz(j)]
                      : 0.0;
      r2[sz(j)] = has_upper[sz(j)]
                      ? target - s[sz(j)] * t[sz(j)] - affine.ds[sz(j)] * affine.dt[sz(j)]
                      : 0.0;
    }
    finite = newton(rp, ru, rd, r1, r2, &direction);
    if (!finite) {
      if (raises < 3) {
        ++raises;
        ++result.regularisation_raises;
        regularisation = std::fmax(regularisation * 1e3, 1e-8);
        continue;
      }
      status = IpmStatus::kNumericalError;
      result.message = "the Newton direction is not finite";
      break;
    }
    double ap = primal_step(direction);
    double ad = dual_step(direction);

    // Gondzio's centrality correctors: aim for a longer step, see which
    // products of the trial point fall outside [0.1, 10] times the target, and
    // correct just those.
    for (Int k = 0; k < options.max_correctors && pairs > 0; ++k) {
      if (std::fmin(ap, ad) >= 1.0 / options.step_fraction) break;
      const double aspiration = 0.1;
      const double trial_p = std::fmin(1.0, 1.5 * std::fmin(1.0, ap) + aspiration);
      const double trial_d = std::fmin(1.0, 1.5 * std::fmin(1.0, ad) + aspiration);
      const double low = 0.1 * target;
      const double high = 10.0 * target;
      for (Int j = 0; j < n; ++j) {
        r1[sz(j)] = 0.0;
        r2[sz(j)] = 0.0;
        if (has_lower[sz(j)]) {
          const double product_j = (v[sz(j)] + trial_p * direction.dv[sz(j)]) *
                                   (z[sz(j)] + trial_d * direction.dz[sz(j)]);
          if (product_j < low) {
            r1[sz(j)] = low - product_j;
          } else if (product_j > high) {
            r1[sz(j)] = std::fmax(high - product_j, -high);
          }
        }
        if (has_upper[sz(j)]) {
          const double product_j = (s[sz(j)] + trial_p * direction.ds[sz(j)]) *
                                   (t[sz(j)] + trial_d * direction.dt[sz(j)]);
          if (product_j < low) {
            r2[sz(j)] = low - product_j;
          } else if (product_j > high) {
            r2[sz(j)] = std::fmax(high - product_j, -high);
          }
        }
      }
      if (!newton(zero_m, zero_n, zero_n, r1, r2, &correction)) break;
      for (Int j = 0; j < n; ++j) {
        correction.dv[sz(j)] += direction.dv[sz(j)];
        correction.ds[sz(j)] += direction.ds[sz(j)];
        correction.dz[sz(j)] += direction.dz[sz(j)];
        correction.dt[sz(j)] += direction.dt[sz(j)];
      }
      for (Int i = 0; i < m; ++i) correction.dy[sz(i)] += direction.dy[sz(i)];
      const double ap_new = primal_step(correction);
      const double ad_new = dual_step(correction);
      const double before = std::fmin(1.0, ap) + std::fmin(1.0, ad);
      const double after = std::fmin(1.0, ap_new) + std::fmin(1.0, ad_new);
      if (after < before + 0.01) break;
      std::swap(direction, correction);
      ap = ap_new;
      ad = ad_new;
      ++result.correctors;
    }

    ap = std::fmin(1.0, options.step_fraction * ap);
    ad = std::fmin(1.0, options.step_fraction * ad);
    if (ap < 1e-12 && ad < 1e-12) {
      if (raises < 3) {
        ++raises;
        ++result.regularisation_raises;
        regularisation = std::fmax(regularisation * 1e3, 1e-8);
        continue;
      }
      status = IpmStatus::kNumericalError;
      result.message = "the step length has collapsed";
      break;
    }
    for (Int j = 0; j < n; ++j) {
      v[sz(j)] += ap * direction.dv[sz(j)];
      if (has_upper[sz(j)]) {
        s[sz(j)] += ap * direction.ds[sz(j)];
        t[sz(j)] += ad * direction.dt[sz(j)];
      }
      if (has_lower[sz(j)]) z[sz(j)] += ad * direction.dz[sz(j)];
    }
    for (Int i = 0; i < m; ++i) y[sz(i)] += ad * direction.dy[sz(i)];
  }

  result.status = status;
  result.iterations = iteration;
  result.objective = lp.model_objective(result.x);
  result.factor_seconds = factor_time;
  result.solve_seconds = elapsed();
  if (result.message.empty() && status == IpmStatus::kIterationLimit) {
    result.message = "iteration limit; the best point reached is returned";
  }
  return result;
}

namespace {

// Crossover from a seed point, then the simplex to finish: the same
// composition the simplex command's own --crossover path uses (see
// command_simplex in app/anukulan_cli.cpp). A seed that produces no usable
// basis, or a warm run that does not reach an optimum, falls back to a cold
// solve - crossover can save pivots or do nothing, never cost an answer.
SimplexResult finish_from_seed(const StandardLp& lp, const std::vector<double>& seed_x,
                               const std::vector<double>& seed_y,
                               const CrossoverOptions& crossover_options,
                               const SimplexOptions& base_options) {
  const CrossoverResult cross = crossover_basis(lp, seed_x, seed_y, crossover_options);
  SimplexResult warm;
  bool have_warm = false;
  if (cross.ok) {
    SimplexOptions warm_options = base_options;
    warm_options.start_basic = &cross.basic;
    warm_options.start_status = &cross.status;
    warm = solve_lp(lp, warm_options);
    have_warm = true;
  }
  if (!have_warm || warm.status != SimplexStatus::kOptimal) {
    const SimplexResult cold = solve_lp(lp, base_options);
    if (!have_warm || cold.status == SimplexStatus::kOptimal) return cold;
  }
  return warm;
}

}  // namespace

RaceResult solve_lp_race(const Model& original_model, const StandardLp& lp,
                         const PostsolveStack* postsolve, const RaceOptions& options) {
  const auto race_start = std::chrono::steady_clock::now();
  auto elapsed = [&race_start]() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - race_start).count();
  };
  auto remaining = [&elapsed, &options]() {
    return std::fmax(0.0, options.time_limit_seconds - elapsed());
  };

  RaceResult result;
  std::atomic<bool> stop{false};
  std::mutex mutex;
  bool have_winner = false;

  // Records one engine's finish. The first to pass the check - optimal, and
  // still optimal after postsolve is applied and the answer is measured
  // against the model the caller actually handed in - sets `stop`, which is
  // how the other two, mid-solve on their own threads, find out they lost.
  auto record = [&](const std::string& name, const SimplexResult& r, double seconds,
                    Int seed_iterations) {
    RaceEngineOutcome outcome;
    outcome.name = name;
    outcome.ran = true;
    outcome.status = r.status;
    outcome.seconds = seconds;
    outcome.seed_iterations = seed_iterations;
    outcome.simplex_iterations = r.iterations;
    outcome.message = r.message;

    std::vector<double> x = r.x;
    if (!x.empty() && postsolve != nullptr) x = postsolve->apply(x);
    ModelViolation checked;
    if (!x.empty()) checked = measure_violation(original_model, x);
    const bool passes =
        r.status == SimplexStatus::kOptimal && !x.empty() && checked.relative_row_violation <= 1e-6;

    const std::lock_guard<std::mutex> lock(mutex);
    if (passes && !have_winner) {
      have_winner = true;
      outcome.won = true;
      result.winner = name;
      result.status = SimplexStatus::kOptimal;
      result.x = std::move(x);
      result.y = r.y;
      result.objective = r.objective;
      result.row_violation = checked.row_violation;
      result.bound_violation = checked.bound_violation;
      stop.store(true, std::memory_order_relaxed);
    }
    result.engines.push_back(std::move(outcome));
  };

  std::vector<std::thread> threads;
  if (options.use_simplex) {
    threads.emplace_back([&]() {
      const double t0 = elapsed();
      SimplexOptions so = options.simplex_options;
      so.start_basic = nullptr;
      so.start_status = nullptr;
      so.time_limit_seconds = remaining();
      so.stop = &stop;
      const SimplexResult r = solve_dual_simplex(lp, so);
      record("simplex", r, elapsed() - t0, 0);
    });
  }
  if (options.use_ipm) {
    threads.emplace_back([&]() {
      const double t0 = elapsed();
      IpmOptions io = options.ipm_options;
      io.time_limit_seconds = remaining();
      io.stop = &stop;
      const IpmResult ir = solve_ipm(lp, io);
      SimplexOptions so = options.simplex_options;
      so.time_limit_seconds = remaining();
      so.stop = &stop;
      const SimplexResult r = finish_from_seed(lp, ir.x, ir.y, options.crossover_options, so);
      record("ipm", r, elapsed() - t0, ir.iterations);
    });
  }
  if (options.use_pdhg) {
    threads.emplace_back([&]() {
      const double t0 = elapsed();
      PdhgOptions po = options.pdhg_options;
      po.tolerance = options.pdhg_seed_tolerance;
      po.gap_tolerance = options.pdhg_seed_tolerance;
      Int budget = options.pdhg_seed_max_iterations;
      if (options.pdhg_seed_iterations_per_row > 0.0) {
        budget = std::max(budget, static_cast<Int>(options.pdhg_seed_iterations_per_row *
                                                    static_cast<double>(lp.num_rows())));
      }
      if (budget > 0) po.max_iterations = budget;
      po.time_limit_seconds = remaining();
      po.stop = &stop;
      const PdhgResult pr = solve_pdhg(lp, po);
      SimplexOptions so = options.simplex_options;
      so.time_limit_seconds = remaining();
      so.stop = &stop;
      const SimplexResult r = finish_from_seed(lp, pr.x, pr.y, options.crossover_options, so);
      record("pdhg", r, elapsed() - t0, pr.iterations);
    });
  }
  for (std::thread& t : threads) t.join();

  if (!have_winner) {
    // Nobody produced a passing optimum: every engine ran out of clock, or the
    // problem is infeasible or unbounded and reaching that conclusion is not
    // what decides this race. Report whichever outcome is most informative,
    // a definitive proof over a run that simply did not finish.
    auto rank = [](SimplexStatus s) {
      switch (s) {
        case SimplexStatus::kInfeasible:
        case SimplexStatus::kUnbounded:
          return 3;
        case SimplexStatus::kTimeLimit:
        case SimplexStatus::kIterationLimit:
          return 2;
        case SimplexStatus::kOptimal:
          return 1;  // reached, but did not pass the check
        case SimplexStatus::kNumericalError:
          return 0;
      }
      return 0;
    };
    const RaceEngineOutcome* best = nullptr;
    for (const RaceEngineOutcome& outcome : result.engines) {
      if (best == nullptr || rank(outcome.status) > rank(best->status)) best = &outcome;
    }
    if (best != nullptr) {
      result.status = best->status;
      result.message = best->name + ": " + best->message;
    } else {
      result.message = "no engine ran";
    }
  }
  return result;
}

}  // namespace anukulan
