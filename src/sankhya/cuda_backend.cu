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
#include <map>
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

// A group of lanes per row, with the group size chosen from how long the rows
// actually are.
//
// A whole 32-lane warp per row is the textbook CSR-vector kernel and it wastes
// most of the warp when rows are short. The profile says so directly. On
// graph40-40 the same kernel over the same 1,260,900 nonzeros takes 1256 us for
// K, whose rows hold 3.49 nonzeros, and 384 us for K transpose, whose rows hold
// 12.29 - a factor of 3.3 for a factor of 3.5 in occupancy. qap15 shows the
// same thing with the roles swapped: its K transpose has 4.26 per row and takes
// 46.5 us against 19.4 us for K at 15.0. Same code, same nonzero count, only
// the row length differs, in both directions.
//
// So the width comes from the matrix: the smallest power of two at least as
// large as the average row, between 2 and 32. graph40-40's K gets 4 lanes per
// row instead of 32, which puts eight times as many rows in flight per warp.
//
// Bell and Garland, Implementing Sparse Matrix-Vector Multiplication on
// Throughput-Oriented Processors, SC 2009.
template <int kVector>
__global__ void spmv_kernel(Int rows, const Int* __restrict__ start,
                            const Int* __restrict__ index,
                            const double* __restrict__ value,
                            const double* __restrict__ x, double* __restrict__ y) {
  const int lane = static_cast<int>(threadIdx.x) % kVector;
  const int vector_in_block = static_cast<int>(threadIdx.x) / kVector;
  const int vectors_per_block = kBlock / kVector;
  const Int stride = static_cast<Int>(vectors_per_block) * static_cast<Int>(gridDim.x);

  // The loop runs over the block's first row, not over each vector's own row,
  // so its condition is identical for every thread in the block. With several
  // vectors to a warp, letting each leave as its row ran out would have some
  // lanes reach __shfl_down_sync while others had returned, and the full-warp
  // mask below would then be a lie. Rows past the end fall through with a zero
  // partial sum and write nothing.
  for (Int base = static_cast<Int>(blockIdx.x) * vectors_per_block; base < rows;
       base += stride) {
    const Int row = base + vector_in_block;
    double sum = 0.0;
    if (row < rows) {
      const Int begin = start[row];
      const Int end = start[row + 1];
      for (Int k = begin + lane; k < end; k += kVector) sum += value[k] * x[index[k]];
    }

    // kVector divides 32, so a shuffle by less than kVector reaches outside this
    // row only in lanes whose result is discarded: lane 0 accumulates from
    // inside its own vector at every step.
    for (int offset = kVector / 2; offset > 0; offset >>= 1)
      sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0 && row < rows) y[row] = sum;
  }
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

__global__ void blend_kernel(Int n, double a, double* __restrict__ z, double b,
                             const double* __restrict__ anchor) {
  const Int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) z[i] = a * z[i] + b * anchor[i];
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
  Int nnz = 0;
  int vector_width = 32;  // lanes per row, chosen from the average row length
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

  void set_profiling(bool on) const override {
    if (on && ev_start_ == nullptr) {
      cudaEventCreate(&ev_start_);
      cudaEventCreate(&ev_stop_);
    }
    profiling_ = on;
    if (!on) timings_.clear();
  }

  std::string profile_report() const override {
    if (timings_.empty()) return {};
    std::string widths;
    for (const auto& [key, d] : matrices_) {
      (void)key;
      char w[128];
      std::snprintf(w, sizeof(w),
                    "  %8d rows, %9d nonzeros, %5.2f per row -> %2d lanes per row\n",
                    d.rows, d.nnz,
                    d.rows > 0 ? static_cast<double>(d.nnz) / d.rows : 0.0,
                    d.vector_width);
      widths += w;
    }
    double total = 0.0;
    long long launches = 0;
    for (const auto& [name, t] : timings_) {
      total += t.seconds;
      launches += t.calls;
    }
    std::vector<std::pair<std::string, Timing>> sorted(timings_.begin(), timings_.end());
    std::sort(sorted.begin(), sorted.end(),
              [](const auto& a, const auto& b) { return a.second.seconds > b.second.seconds; });
    char line[256];
    std::string out;
    std::snprintf(line, sizeof(line),
                  "%-22s %12s %10s %14s %8s\n",
                  "kernel", "seconds", "share", "launches", "us each");
    out += line;
    for (const auto& [name, t] : sorted) {
      std::snprintf(line, sizeof(line), "%-22s %12.4f %9.1f%% %14lld %8.1f\n",
                    name.c_str(), t.seconds,
                    total > 0.0 ? 100.0 * t.seconds / total : 0.0, t.calls,
                    t.calls > 0 ? 1e6 * t.seconds / static_cast<double>(t.calls) : 0.0);
      out += line;
    }
    std::snprintf(line, sizeof(line), "%-22s %12.4f %9.1f%% %14lld\n",
                  "total in kernels", total, 100.0, launches);
    out += line;
    if (!widths.empty()) out += "\nsparse products:\n" + widths;
    return out;
  }

  ~CudaBackend() override {
    release();
    if (ev_start_) cudaEventDestroy(ev_start_);
    if (ev_stop_) cudaEventDestroy(ev_stop_);
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
    { Scope timer(this, "fill"); fill_kernel<<<blocks(n), kBlock>>>(n, value, target); }
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
    d.nnz = a.nnz();
    d.vector_width = choose_vector_width(a.rows(), a.nnz());
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
    spmv(a, x, y, "spmv K x");
  }

  void multiply_transpose(const SparseMatrix& at, const double* y,
                          double* x) const override {
    spmv(at, y, x, "spmv Kt y");
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
    Scope timer(this, "primal step");
    primal_step_kernel<<<blocks(n), kBlock>>>(n, tau, x, c, kt_y, lower, upper,
                                              x_next, dx, x_bar);
  }

  void dual_step(Int m, Int num_equalities, double sigma, const double* y,
                 const double* q, const double* k_x_bar, const double* k_x,
                 double* y_next, double* dy, double* k_dx) const override {
    if (m <= 0) return;
    Scope timer(this, "dual step");
    dual_step_kernel<<<blocks(m), kBlock>>>(m, num_equalities, sigma, y, q, k_x_bar,
                                            k_x, y_next, dy, k_dx);
  }

  void advance_kx(Int m, const double* k_x_bar, double* k_x) const override {
    if (m > 0) {
      Scope timer(this, "advance Kx");
      advance_kx_kernel<<<blocks(m), kBlock>>>(m, k_x_bar, k_x);
    }
  }

  void accumulate(Int n, double weight, const double* v, double* sum) const override {
    if (n > 0) {
      Scope timer(this, "accumulate");
      accumulate_kernel<<<blocks(n), kBlock>>>(n, weight, v, sum);
    }
  }

  void scale_into(Int n, double weight, const double* sum, double* out) const override {
    if (n > 0) {
      Scope timer(this, "scale into");
      scale_into_kernel<<<blocks(n), kBlock>>>(n, 1.0 / weight, sum, out);
    }
  }

  void blend(Int n, double a, double* z, double b,
             const double* anchor) const override {
    if (n > 0) {
      Scope timer(this, "halpern blend");
      blend_kernel<<<blocks(n), kBlock>>>(n, a, z, b, anchor);
    }
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
    {
      Scope timer(this, "step-size terms");
      step_terms_kernel<<<grid, kBlock>>>(n, m, dx, dy, k_dx, terms_partial_, grid);
      finish_terms_kernel<<<3, kBlock>>>(grid, terms_partial_, terms_);
    }
    double host[3] = {0.0, 0.0, 0.0};
    {
      Scope timer(this, "step-size copy back");
      cuda_check(cudaMemcpy(host, terms_, sizeof(double) * 3, cudaMemcpyDeviceToHost),
                 "step size terms");
    }
    // The absolute value matches the reference implementation exactly; see the
    // comment on LinAlgBackend::step_size_terms for why it belongs here.
    *interaction = std::fabs(host[0]);
    *movement = omega * host[1] + host[2] / omega;
  }

 private:
  struct Timing {
    double seconds = 0.0;
    long long calls = 0;
  };

  // Wraps one launch. Does nothing at all unless profiling is on, so the
  // ordinary path pays a branch and no synchronisation.
  struct Scope {
    const CudaBackend* owner;
    const char* name;
    Scope(const CudaBackend* o, const char* n) : owner(o), name(n) {
      if (owner->profiling_) cudaEventRecord(owner->ev_start_);
    }
    ~Scope() {
      if (!owner->profiling_) return;
      cudaEventRecord(owner->ev_stop_);
      cudaEventSynchronize(owner->ev_stop_);
      float ms = 0.0f;
      cudaEventElapsedTime(&ms, owner->ev_start_, owner->ev_stop_);
      Timing& t = owner->timings_[name];
      t.seconds += static_cast<double>(ms) / 1000.0;
      t.calls += 1;
    }
  };

  static constexpr int kMaxBlocks = 512;

  static Int blocks(Int n) {
    const Int b = static_cast<Int>((static_cast<long long>(n) + kBlock - 1) / kBlock);
    return b > 0 ? b : 1;
  }

  // The smallest power of two at least as wide as the average row, held between
  // 2 and 32. A row shorter than its vector leaves lanes idle for that row; a
  // row longer than it simply loops, which costs nothing. So rounding up is the
  // safe direction and the average is the right statistic to round from.
  static int choose_vector_width(Int rows, Int nnz) {
    if (rows <= 0 || nnz <= 0) return 2;
    const double per_row = static_cast<double>(nnz) / static_cast<double>(rows);
    int width = 2;
    while (width < 32 && static_cast<double>(width) < per_row) width *= 2;
    return width;
  }

  void spmv(const SparseMatrix& a, const double* x, double* y, const char* label) const {
    prepare(a);
    const DeviceMatrix& d = matrices_.at(&a);
    const int vectors_per_block = kBlock / d.vector_width;
    Int grid = (d.rows + vectors_per_block - 1) / vectors_per_block;
    if (grid < 1) grid = 1;
    if (grid > 65535) grid = 65535;  // the kernel strides, so a cap is fine
    Scope timer(this, label);
    switch (d.vector_width) {
      case 2:
        spmv_kernel<2><<<grid, kBlock>>>(d.rows, d.start, d.index, d.value, x, y);
        break;
      case 4:
        spmv_kernel<4><<<grid, kBlock>>>(d.rows, d.start, d.index, d.value, x, y);
        break;
      case 8:
        spmv_kernel<8><<<grid, kBlock>>>(d.rows, d.start, d.index, d.value, x, y);
        break;
      case 16:
        spmv_kernel<16><<<grid, kBlock>>>(d.rows, d.start, d.index, d.value, x, y);
        break;
      default:
        spmv_kernel<32><<<grid, kBlock>>>(d.rows, d.start, d.index, d.value, x, y);
        break;
    }
  }

  template <bool kMaximum>
  double reduce(const double* a, const double* b, Int n) const {
    const int grid = static_cast<int>(std::min<Int>(kMaxBlocks, blocks(n)));
    {
      Scope timer(this, kMaximum ? "reduce max" : "reduce sum");
      reduce_kernel<kMaximum><<<grid, kBlock>>>(n, a, b, partial_);
      cuda_check(cudaMemcpy(host_partial_.data(), partial_, sizeof(double) * sz(grid),
                            cudaMemcpyDeviceToHost), "reduce");
    }
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

  mutable bool profiling_ = false;
  mutable cudaEvent_t ev_start_ = nullptr;
  mutable cudaEvent_t ev_stop_ = nullptr;
  mutable std::map<std::string, Timing> timings_;
};

}  // namespace

const LinAlgBackend& cuda_backend() {
  static const CudaBackend backend;
  return backend;
}

}  // namespace sankhya
