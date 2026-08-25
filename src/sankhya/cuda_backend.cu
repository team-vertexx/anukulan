// CUDA implementation of LinAlgBackend.
//
// Everything here is hand written. cuSPARSE would do the sparse products, and
// would probably do them faster, but the problem statement asks for a solver
// built from mathematical foundation rather than assembled from libraries, and
// a CSR product is forty lines. Being able to open this file and explain every
// line of it is worth more here than the last few percent of bandwidth.
//
// The five kernels that matter are the two products and the two fused steps,
// plus the reductions. The fused steps exist because a backend offering only
// axpy and dot would force a dozen launches per iteration; each of these reads
// its vectors once and writes them once.

#include <cuda_runtime.h>

#include <cstdio>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "sankhya/backend.hpp"

namespace sankhya {
namespace {

constexpr int kBlock = 256;

void cuda_check(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string("cuda: ") + what + ": " +
                             cudaGetErrorString(status));
  }
}

// One warp per row. Rows in these problems are short - a handful of nonzeros -
// so a whole warp per row wastes lanes, but it keeps the loads coalesced and
// avoids the divergence a thread-per-row layout suffers when row lengths differ.
__global__ void spmv_kernel(Int rows, const Int* __restrict__ start,
                            const Int* __restrict__ index,
                            const double* __restrict__ value,
                            const double* __restrict__ x, double* __restrict__ y) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (warp >= rows) return;

  const Int begin = start[warp];
  const Int end = start[warp + 1];
  double sum = 0.0;
  for (Int k = begin + lane; k < end; k += 32) sum += value[k] * x[index[k]];

  for (int offset = 16; offset > 0; offset >>= 1)
    sum += __shfl_down_sync(0xffffffffu, sum, offset);
  if (lane == 0) y[warp] = sum;
}

__global__ void primal_step_kernel(Int n, double tau, const double* __restrict__ x,
                                   const double* __restrict__ c,
                                   const double* __restrict__ kt_y,
                                   const double* __restrict__ lower,
                                   const double* __restrict__ upper,
                                   double* __restrict__ x_next,
                                   double* __restrict__ dx,
                                   double* __restrict__ x_bar) {
  const Int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= n) return;
  double v = x[j] - tau * (c[j] - kt_y[j]);
  v = fmin(fmax(v, lower[j]), upper[j]);
  const double d = v - x[j];
  x_next[j] = v;
  dx[j] = d;
  x_bar[j] = v + d;
}

__global__ void dual_step_kernel(Int m, Int num_equalities, double sigma,
                                 const double* __restrict__ y,
                                 const double* __restrict__ q,
                                 const double* __restrict__ k_x_bar,
                                 const double* __restrict__ k_x,
                                 double* __restrict__ y_next,
                                 double* __restrict__ dy,
                                 double* __restrict__ k_dx) {
  const Int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= m) return;
  double v = y[i] + sigma * (q[i] - k_x_bar[i]);
  if (i >= num_equalities) v = fmax(v, 0.0);
  y_next[i] = v;
  dy[i] = v - y[i];
  k_dx[i] = 0.5 * (k_x_bar[i] - k_x[i]);
}

__global__ void advance_kx_kernel(Int m, const double* __restrict__ k_x_bar,
                                  double* __restrict__ k_x) {
  const Int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < m) k_x[i] = 0.5 * (k_x_bar[i] + k_x[i]);
}

__global__ void accumulate_kernel(Int n, double weight, const double* __restrict__ v,
                                  double* __restrict__ sum) {
  const Int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) sum[i] += weight * v[i];
}

__global__ void scale_into_kernel(Int n, double inv_weight,
                                  const double* __restrict__ sum,
                                  double* __restrict__ out) {
  const Int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = sum[i] * inv_weight;
}

// Reductions. Two of them: a sum for dot products and a maximum for the infinity
// norm. Both use the same block-then-finish shape.
template <bool kMaximum>
__global__ void reduce_kernel(Int n, const double* __restrict__ a,
                              const double* __restrict__ b,
                              double* __restrict__ partial) {
  __shared__ double shared[kBlock];
  const Int stride = blockDim.x * gridDim.x;
  double acc = 0.0;
  for (Int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    const double v = kMaximum ? fabs(a[i]) : a[i] * (b ? b[i] : a[i]);
    acc = kMaximum ? fmax(acc, v) : acc + v;
  }
  shared[threadIdx.x] = acc;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      shared[threadIdx.x] = kMaximum
                                ? fmax(shared[threadIdx.x], shared[threadIdx.x + s])
                                : shared[threadIdx.x] + shared[threadIdx.x + s];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) partial[blockIdx.x] = shared[0];
}

struct DeviceMatrix {
  Int rows = 0;
  Int* start = nullptr;
  Int* index = nullptr;
  double* value = nullptr;
};

// Device-side scratch that the solver's vectors are staged through. The
// interface hands us host pointers, so each call copies in and out. That is the
// honest cost of keeping one algorithm for both backends, and it is why the
// steps are fused: fewer calls, fewer transfers.
class CudaBackend final : public LinAlgBackend {
 public:
  CudaBackend() {
    int count = 0;
    cuda_check(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
    if (count == 0) throw std::runtime_error("cuda: no device");
    cuda_check(cudaMalloc(&partial_, sizeof(double) * kMaxBlocks), "partial");
    host_partial_.resize(kMaxBlocks);
  }

  ~CudaBackend() override {
    release();
    if (partial_) cudaFree(partial_);
    for (double* p : pool_) cudaFree(p);
  }

  std::string name() const override { return "cuda"; }

  void prepare(const SparseMatrix& a) const override {
    if (matrices_.count(&a)) return;
    DeviceMatrix d;
    d.rows = a.rows();
    const std::size_t start_bytes = sizeof(Int) * (sz(a.rows()) + 1);
    const std::size_t nnz_bytes_i = sizeof(Int) * sz(a.nnz());
    const std::size_t nnz_bytes_d = sizeof(double) * sz(a.nnz());
    cuda_check(cudaMalloc(&d.start, start_bytes), "matrix start");
    cuda_check(cudaMalloc(&d.index, nnz_bytes_i ? nnz_bytes_i : 1), "matrix index");
    cuda_check(cudaMalloc(&d.value, nnz_bytes_d ? nnz_bytes_d : 1), "matrix value");
    cuda_check(cudaMemcpy(d.start, a.start().data(), start_bytes,
                          cudaMemcpyHostToDevice), "copy start");
    if (a.nnz() > 0) {
      cuda_check(cudaMemcpy(d.index, a.index().data(), nnz_bytes_i,
                            cudaMemcpyHostToDevice), "copy index");
      cuda_check(cudaMemcpy(d.value, a.value().data(), nnz_bytes_d,
                            cudaMemcpyHostToDevice), "copy value");
    }
    matrices_[&a] = d;
  }

  void release() const override {
    for (auto& entry : matrices_) {
      cudaFree(entry.second.start);
      cudaFree(entry.second.index);
      cudaFree(entry.second.value);
    }
    matrices_.clear();
  }

  void multiply(const SparseMatrix& a, const double* x, double* y) const override {
    spmv(a, x, y, a.cols(), a.rows());
  }

  void multiply_transpose(const SparseMatrix& at, const double* y,
                          double* x) const override {
    spmv(at, y, x, at.cols(), at.rows());
  }

  double dot(const double* a, const double* b, Int n) const override {
    if (n <= 0) return 0.0;
    double* da = stage(a, n, 0);
    double* db = stage(b, n, 1);
    return reduce<false>(da, db, n);
  }

  double two_norm(const double* a, Int n) const override {
    return std::sqrt(dot(a, a, n));
  }

  double inf_norm(const double* a, Int n) const override {
    if (n <= 0) return 0.0;
    double* da = stage(a, n, 0);
    return reduce<true>(da, nullptr, n);
  }

  void primal_step(Int n, double tau, const double* x, const double* c,
                   const double* kt_y, const double* lower, const double* upper,
                   double* x_next, double* dx, double* x_bar) const override {
    if (n <= 0) return;
    double* dx_in = stage(x, n, 0);
    double* dc = stage(c, n, 1);
    double* dk = stage(kt_y, n, 2);
    double* dl = stage(lower, n, 3);
    double* du = stage(upper, n, 4);
    double* o1 = buffer(n, 5);
    double* o2 = buffer(n, 6);
    double* o3 = buffer(n, 7);
    primal_step_kernel<<<blocks(n), kBlock>>>(n, tau, dx_in, dc, dk, dl, du, o1, o2, o3);
    fetch(o1, x_next, n);
    fetch(o2, dx, n);
    fetch(o3, x_bar, n);
  }

  void dual_step(Int m, Int num_equalities, double sigma, const double* y,
                 const double* q, const double* k_x_bar, const double* k_x,
                 double* y_next, double* dy, double* k_dx) const override {
    if (m <= 0) return;
    double* dy_in = stage(y, m, 0);
    double* dq = stage(q, m, 1);
    double* dkb = stage(k_x_bar, m, 2);
    double* dkx = stage(k_x, m, 3);
    double* o1 = buffer(m, 4);
    double* o2 = buffer(m, 5);
    double* o3 = buffer(m, 6);
    dual_step_kernel<<<blocks(m), kBlock>>>(m, num_equalities, sigma, dy_in, dq, dkb,
                                            dkx, o1, o2, o3);
    fetch(o1, y_next, m);
    fetch(o2, dy, m);
    fetch(o3, k_dx, m);
  }

  void advance_kx(Int m, const double* k_x_bar, double* k_x) const override {
    if (m <= 0) return;
    double* a = stage(k_x_bar, m, 0);
    double* b = stage(k_x, m, 1);
    advance_kx_kernel<<<blocks(m), kBlock>>>(m, a, b);
    fetch(b, k_x, m);
  }

  void accumulate(Int n, double weight, const double* v, double* sum) const override {
    if (n <= 0) return;
    double* dv = stage(v, n, 0);
    double* ds = stage(sum, n, 1);
    accumulate_kernel<<<blocks(n), kBlock>>>(n, weight, dv, ds);
    fetch(ds, sum, n);
  }

  void scale_into(Int n, double weight, const double* sum, double* out) const override {
    if (n <= 0) return;
    double* ds = stage(sum, n, 0);
    double* o = buffer(n, 1);
    scale_into_kernel<<<blocks(n), kBlock>>>(n, 1.0 / weight, ds, o);
    fetch(o, out, n);
  }

  double weighted_norm_squared(Int n, Int m, const double* dx, const double* dy,
                               double omega) const override {
    return omega * dot(dx, dx, n) + dot(dy, dy, m) / omega;
  }

 private:
  static constexpr int kMaxBlocks = 512;
  static constexpr int kSlots = 10;

  static Int blocks(Int n) {
    return static_cast<Int>((static_cast<long long>(n) + kBlock - 1) / kBlock);
  }

  double* buffer(Int n, int slot) const {
    const std::size_t want = sz(n);
    if (pool_.size() <= static_cast<std::size_t>(slot)) {
      pool_.resize(static_cast<std::size_t>(kSlots), nullptr);
      sizes_.resize(static_cast<std::size_t>(kSlots), 0);
    }
    if (sizes_[sz(slot)] < want) {
      if (pool_[sz(slot)]) cudaFree(pool_[sz(slot)]);
      cuda_check(cudaMalloc(&pool_[sz(slot)], sizeof(double) * want), "buffer");
      sizes_[sz(slot)] = want;
    }
    return pool_[sz(slot)];
  }

  double* stage(const double* host, Int n, int slot) const {
    double* device = buffer(n, slot);
    cuda_check(cudaMemcpy(device, host, sizeof(double) * sz(n),
                          cudaMemcpyHostToDevice), "stage");
    return device;
  }

  void fetch(const double* device, double* host, Int n) const {
    cuda_check(cudaMemcpy(host, device, sizeof(double) * sz(n),
                          cudaMemcpyDeviceToHost), "fetch");
  }

  void spmv(const SparseMatrix& a, const double* x, double* y, Int in, Int out) const {
    prepare(a);
    const DeviceMatrix& d = matrices_.at(&a);
    double* dx = stage(x, in, 8);
    double* dy = buffer(out, 9);
    const Int warps_per_block = kBlock / 32;
    const Int grid = (out + warps_per_block - 1) / warps_per_block;
    spmv_kernel<<<grid > 0 ? grid : 1, kBlock>>>(d.rows, d.start, d.index, d.value,
                                                 dx, dy);
    fetch(dy, y, out);
  }

  template <bool kMaximum>
  double reduce(const double* a, const double* b, Int n) const {
    const int grid = static_cast<int>(std::min<Int>(kMaxBlocks, blocks(n)));
    reduce_kernel<kMaximum><<<grid, kBlock>>>(n, a, b, partial_);
    cuda_check(cudaMemcpy(host_partial_.data(), partial_, sizeof(double) * sz(grid),
                          cudaMemcpyDeviceToHost), "reduce");
    double acc = 0.0;
    for (int i = 0; i < grid; ++i) {
      acc = kMaximum ? std::fmax(acc, host_partial_[sz(i)])
                     : acc + host_partial_[sz(i)];
    }
    return acc;
  }

  mutable std::unordered_map<const SparseMatrix*, DeviceMatrix> matrices_;
  mutable std::vector<double*> pool_;
  mutable std::vector<std::size_t> sizes_;
  double* partial_ = nullptr;
  mutable std::vector<double> host_partial_;
};

}  // namespace

const LinAlgBackend& cuda_backend() {
  static const CudaBackend backend;
  return backend;
}

}  // namespace sankhya
