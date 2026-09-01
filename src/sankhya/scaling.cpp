#include "sankhya/scaling.hpp"

#include <cmath>

namespace sankhya {
namespace {

// Largest / smallest nonzero entry of a norm vector. Empty rows and columns are
// skipped: they carry no scale information and would make the ratio infinite.
double spread(const std::vector<double>& norms) {
  double lo = kInf;
  double hi = 0.0;
  for (const double v : norms) {
    if (v <= 0.0) continue;
    lo = std::fmin(lo, v);
    hi = std::fmax(hi, v);
  }
  if (hi == 0.0) return 1.0;
  return hi / lo;
}

void norms_of(const StandardLp& lp, Norm norm, std::vector<double>* rows,
              std::vector<double>* cols) {
  rows->assign(sz(lp.k.rows()), 0.0);
  cols->assign(sz(lp.k.cols()), 0.0);
  lp.k.row_norms(norm, rows->data());
  lp.k.col_norms(norm, cols->data());
}

// One equilibration sweep: divide row i by sqrt(row norm) and column j by
// sqrt(column norm). An empty row or column is left alone.
void equilibrate(StandardLp* lp, Norm norm, Scaling* accumulated) {
  std::vector<double> row_norm;
  std::vector<double> col_norm;
  norms_of(*lp, norm, &row_norm, &col_norm);

  std::vector<double> d1(row_norm.size());
  std::vector<double> d2(col_norm.size());
  for (std::size_t i = 0; i < row_norm.size(); ++i)
    d1[i] = (row_norm[i] > 0.0) ? 1.0 / std::sqrt(row_norm[i]) : 1.0;
  for (std::size_t j = 0; j < col_norm.size(); ++j)
    d2[j] = (col_norm[j] > 0.0) ? 1.0 / std::sqrt(col_norm[j]) : 1.0;

  lp->k.scale_rows(d1.data());
  lp->k.scale_cols(d2.data());

  for (std::size_t i = 0; i < d1.size(); ++i) {
    lp->q[i] *= d1[i];
    accumulated->row_scale[i] *= d1[i];
  }
  for (std::size_t j = 0; j < d2.size(); ++j) {
    lp->c[j] *= d2[j];
    // x = D2 x_tilde, so a smaller d2 means the scaled variable is larger, and
    // the bounds move the other way. Infinite bounds stay infinite.
    lp->lower[j] /= d2[j];
    lp->upper[j] /= d2[j];
    accumulated->col_scale[j] *= d2[j];
  }
}

}  // namespace

void Scaling::unscale_primal(const std::vector<double>& scaled,
                             std::vector<double>* out) const {
  out->resize(col_scale.size());
  for (std::size_t j = 0; j < col_scale.size(); ++j) (*out)[j] = col_scale[j] * scaled[j];
}

void Scaling::unscale_dual(const std::vector<double>& scaled,
                           std::vector<double>* out) const {
  out->resize(row_scale.size());
  for (std::size_t i = 0; i < row_scale.size(); ++i) (*out)[i] = row_scale[i] * scaled[i];
}

ScalingReport scale_lp(StandardLp* lp, const ScalingOptions& options) {
  ScalingReport report;
  report.scaling.row_scale.assign(sz(lp->k.rows()), 1.0);
  report.scaling.col_scale.assign(sz(lp->k.cols()), 1.0);

  std::vector<double> row_norm;
  std::vector<double> col_norm;
  norms_of(*lp, Norm::kInfinity, &row_norm, &col_norm);
  report.row_spread_before = spread(row_norm);
  report.col_spread_before = spread(col_norm);

  for (int iteration = 0; iteration < options.ruiz_iterations; ++iteration) {
    equilibrate(lp, Norm::kInfinity, &report.scaling);
  }
  if (options.pock_chambolle) {
    equilibrate(lp, Norm::kOne, &report.scaling);
  }

  norms_of(*lp, Norm::kInfinity, &row_norm, &col_norm);
  report.row_spread_after = spread(row_norm);
  report.col_spread_after = spread(col_norm);

  // The transpose is cached for the solver, so it has to follow the matrix.
  lp->kt = lp->k.transpose();
  return report;
}

}  // namespace sankhya
