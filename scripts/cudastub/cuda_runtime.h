// Stub CUDA runtime, used only to type-check cuda_backend.cu on a machine with
// no nvcc. It is never compiled into anything that runs.
#pragma once
#include <cstddef>
#include <cstring>
#define __global__
#define __device__
#define __host__
#define __shared__
struct uint3_t { unsigned int x, y, z; };
extern uint3_t blockIdx, threadIdx, blockDim, gridDim;
inline void __syncthreads() {}
typedef int cudaError_t;
enum { cudaSuccess = 0 };
typedef enum { cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost,
               cudaMemcpyDeviceToDevice } cudaMemcpyKind;
inline cudaError_t cudaMalloc(void** p, std::size_t n) { *p = ::operator new(n); return 0; }
template <class T> inline cudaError_t cudaMalloc(T** p, std::size_t n) {
  *p = static_cast<T*>(::operator new(n)); return 0; }
inline cudaError_t cudaFree(void* p) { ::operator delete(p); return 0; }
inline cudaError_t cudaMemcpy(void* d, const void* s, std::size_t n, cudaMemcpyKind) {
  std::memcpy(d, s, n); return 0; }
inline cudaError_t cudaMemset(void* d, int v, std::size_t n) {
  std::memset(d, v, n); return 0; }
inline cudaError_t cudaGetDeviceCount(int* n) { *n = 1; return 0; }
inline const char* cudaGetErrorString(cudaError_t) { return "stub"; }
template <class T> inline T __shfl_down_sync(unsigned, T v, unsigned) { return v; }
