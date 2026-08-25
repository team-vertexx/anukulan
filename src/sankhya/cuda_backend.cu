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

#include <algorithm>
#include <cmath>
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

__global__ void fill_kernel(Int n, double value, double* __restrict__ target) {
  const Int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) target[i] = value;
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


// The three inner products the adaptive step size needs, in one launch.
//
// Separately these are three reduce() calls, and each ends in a device-to-host
// copy that synchronises the whole pipeline. The bytes are trivial - a few
// kilobytes of block partials - but the stall is not, and it happens on every
// iteration. Fused, the block partials for all three land in one array, a second
// kernel finishes them on the device, and exactly three doubles cross the bus.
//
// The block partials are summed in a different order than three separate
// reductions would sum them, because the grid is now sized from max(n, m)
// rather than from each length on its own. Floating point addition is not
// associative, so the last bits move. That is the same class of difference the
// build already sees from FMA contraction, and it is why iteration counts are
// never used as a correctness gate here - objectives against published optima
// are.
__global__ void step_terms_kernel(Int n, Int m, const double* __restrict__ dx,
                                  const double* __restrict__ dy,
                                  const double* __restrict__ k_dx,
                                  double* __restrict__ partial, int grid) {
  __shared__ double shared[3][kBlock];
  const Int stride = blockDim.x * gridDim.x;
  const Int start = blockIdx.x * blockDim.x + threadIdx.x;

  double interaction = 0.0, dy_dy = 0.0, dx_dx = 0.0;
  for (Int i = start; i < m; i += stride) {
    const double v = dy[i];
    interaction += v * k_dx[i];
    dy_dy += v * v;
  }
  for (Int i = start; i < n; i += stride) {
    const double v = dx[i];
    dx_dx += v * v;
  }

  shared[0][threadIdx.x] = interaction;
  shared[1][threadIdx.x] = dx_dx;
  shared[2][threadIdx.x] = dy_dy;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      shared[0][threadIdx.x] += shared[0][threadIdx.x + s];
      shared[1][threadIdx.x] += shared[1][threadIdx.x + s];
      shared[2][threadIdx.x] += shared[2][threadIdx.x + s];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    partial[blockIdx.x] = shared[0][0];
    partial[grid + blockIdx.x] = shared[1][0];
    partial[2 * grid + blockIdx.x] = shared[2][0];
  }
}

// One block per quantity, so the whole finish is a single launch of three
// blocks and the result never leaves the device until it is three doubles.
__global__ void finish_terms_kernel(int grid, const double* __restrict__ partial,
                                    double* __restrict__ out) {
  __shared__ double shared[kBlock];
  const int which = blockIdx.x;
  double acc = 0.0;
  for (int i = threadIdx.x; i < grid; i += blockDim.x)
    acc += partial[which * grid + i];
  shared[threadIdx.x] = acc;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) shared[threadIdx.x] += shared[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) out[which] = shared[0];
}

struct DeviceMatrix {
  Int rows = 0;
  Int* start = nullptr;
  Int* index = nullptr;
  double* value = nullptr;
};

// The vectors the solver works on live here, allocated through the interface
// and never copied back until the solver asks. An earlier version staged every
// argument across the bus on every call, which is about ten transfers per
// iteration, and measured slower on a T4 than the CPU it was meant to
// accelerate. The kernels were not the problem.
class CudaBackend final : public LinAlgBackend {
 public:
  CudaBackend() {
    int count = 0;
    cuda_check(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
    if (count == 0) throw std::runtime_error("cuda: no device");
    cuda_check(cudaMalloc(&partial_, sizeof(double) * kMaxBlocks), "partial");
    host_partial_.resize(kMaxBlocks);
    cuda_check(cudaMalloc(&terms_partial_, sizeof(double) * 3 * kMaxBlocks),
               "step term partials");
    cuda_check(cudaMalloc(&terms_, sizeof(double) * 3), "step terms");
  }

  ~CudaBackend() override {
    release();
    if (partial_) cudaFree(partial_);
    if (terms_partial_) cudaFree(terms_partial_);
    if (terms_) cudaFree(terms_);
  }

  std::string name() const override { return "cuda"; }

  double* allocate(Int n) const override {
    if (n <= 0) return nullptr;
    double* p = nullptr;
    cuda_check(cudaMalloc(&p, sizeof(double) * sz(n)), "allocate");
    cuda_check(cudaMemset(p, 0, sizeof(double) * sz(n)), "allocate zero");
    return p;
  }

  void deallocate(double* p) const override {
    if (p) cudaFree(p);
  }

  void upload(const double* host, double* target, Int n) const override {
    if (n <= 0) return;
    cuda_check(cudaMemcpy(target, host, sizeof(double) * sz(n),
                          cudaMemcpyHostToDevice), "upload");
  }

  void download(const double* source, double* host, Int n) const override {
    if (n <= 0) return;
    cuda_check(cudaMemcpy(host, source, sizeof(double) * sz(n),
                          cudaMemcpyDeviceToHost), "download");
  }

  void fill(double* target, Int n, double value) const override {
    if (n <= 0) return;
    if (value == 0.0) {
      cuda_check(cudaMemset(target, 0, sizeof(double) * sz(n)), "fill zero");
      return;
    }
    fill_kernel<<<blocks(n), kBlock>>>(n, value, target);
  }

  void copy(const double* source, double* target, Int n) const override {
    if (n <= 0) return;
    cuda_check(cudaMemcpy(target, source, sizeof(double) * sz(n),
                          cudaMemcpyDeviceToDevice), "copy");
  }

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
    spmv(a, x, y);
  }

  void multiply_transpose(const SparseMatrix& at, const double* y,
                          double* x) const override {
    spmv(at, y, x);
  }

  double dot(const double* a, const double* b, Int n) const override {
    return n > 0 ? reduce<false>(a, b, n) : 0.0;
  }

  double two_norm(const double* a, Int n) const override {
    return std::sqrt(dot(a, a, n));
  }

  double inf_norm(const double* a, Int n) const override {
    return n > 0 ? reduce<true>(a, nullptr, n) : 0.0;
  }

  void primal_step(Int n, double tau, const double* x, const double* c,
                   const double* kt_y, const double* lower, const double* upper,
                   double* x_next, double* dx, double* x_bar) const override {
    if (n <= 0) return;
    primal_step_kernel<<<blocks(n), kBlock>>>(n, tau, x, c, kt_y, lower, upper,
                                              x_next, dx, x_bar);
  }

  void dual_step(Int m, Int num_equalities, double sigma, const double* y,
                 const double* q, const double* k_x_bar, const double* k_x,
                 double* y_next, double* dy, double* k_dx) const override {
    if (m <= 0) return;
    dual_step_kernel<<<blocks(m), kBlock>>>(m, num_equalities, sigma, y, q, k_x_bar,
                                            k_x, y_next, dy, k_dx);
  }

  void advance_kx(Int m, const double* k_x_bar, double* k_x) const override {
    if (m > 0) advance_kx_kernel<<<blocks(m), kBlock>>>(m, k_x_bar, k_x);
  }

  void accumulate(Int n, double weight, const double* v, double* sum) const override {
    if (n > 0) accumulate_kernel<<<blocks(n), kBlock>>>(n, weight, v, sum);
  }

  void scale_into(Int n, double weight, const double* sum, double* out) const override {
    if (n > 0) scale_into_kernel<<<blocks(n), kBlock>>>(n, 1.0 / weight, sum, out);
  }

  double weighted_norm_squared(Int n, Int m, const double* dx, const double* dy,
                               double omega) const override {
    return omega * dot(dx, dx, n) + dot(dy, dy, m) / omega;
  }

  void step_size_terms(Int n, Int m, const double* dx, const double* dy,
                       const double* k_dx, double omega, double* interaction,
                       double* movement) const override {
    const Int longest = n > m ? n : m;
    if (longest <= 0) {
      *interaction = 0.0;
      *movement = 0.0;
      return;
    }
    const int grid = static_cast<int>(std::min<Int>(kMaxBlocks, blocks(longest)));
    step_terms_kernel<<<grid, kBlock>>>(n, m, dx, dy, k_dx, terms_partial_, grid);
    finish_terms_kernel<<<3, kBlock>>>(grid, terms_partial_, terms_);
    double host[3] = {0.0, 0.0, 0.0};
    cuda_check(cudaMemcpy(host, terms_, sizeof(double) * 3, cudaMemcpyDeviceToHost),
               "step size terms");
    // The absolute value matches the reference implementation exactly; see the
    // comment on LinAlgBackend::step_size_terms for why it belongs here.
    *interaction = std::fabs(host[0]);
    *movement = omega * host[1] + host[2] / omega;
  }

 private:
  static constexpr int kMaxBlocks = 512;

  static Int blocks(Int n) {
    const Int b = static_cast<Int>((static_cast<long long>(n) + kBlock - 1) / kBlock);
    return b > 0 ? b : 1;
  }

  void spmv(const SparseMatrix& a, const double* x, double* y) const {
    prepare(a);
    const DeviceMatrix& d = matrices_.at(&a);
    const Int warps_per_block = kBlock / 32;
    const Int grid = (d.rows + warps_per_block - 1) / warps_per_block;
    spmv_kernel<<<grid > 0 ? grid : 1, kBlock>>>(d.rows, d.start, d.index, d.value,
                                                 x, y);
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
  double* partial_ = nullptr;
  mutable std::vector<double> host_partial_;
  double* terms_partial_ = nullptr;  // 3 x kMaxBlocks block sums
  double* terms_ = nullptr;          // the three finished values
};

}  // namespace

const LinAlgBackend& cuda_backend() {
  static const CudaBackend backend;
  return backend;
}

}  // namespace sankhya
