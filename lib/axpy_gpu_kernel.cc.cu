/*
这里是对axpy的GPU实现
*/

#include "axpy.h"
#include "axpy_gpu_kernel.h"
#include "axpy_gpu_kernel_helpers.h"

namespace axpy_jax {

namespace {

template <typename T>
__global__ void axpy_kernel(const T *a, const T *x, const T *y, T *result, std::int64_t size) {
    for (std::int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size;
       idx += blockDim.x * gridDim.x) {
        compute_axpy<T>(a[idx], x[idx], y[idx], result + idx);
    }
}

void ThrowIfError(cudaError_t error) {
  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }
}

template <typename T>
inline void apply_axpy(cudaStream_t stream, void **buffers, const char *opaque,
                         std::size_t opaque_len) {
  const AxpyDescriptor &d = *UnpackDescriptor<AxpyDescriptor>(opaque, opaque_len);
  const std::int64_t size = d.size;

  const T *a = reinterpret_cast<const T *>(buffers[0]);
  const T *x = reinterpret_cast<const T *>(buffers[1]);
  const T *y = reinterpret_cast<const T *>(buffers[2]);

  T *result = reinterpret_cast<T *>(buffers[3]);

  // 可以考虑如何设置这些使得分配的软件更适合算法
  const int block_dim = 128;
  const int grid_dim = std::min<int>(1024, (size + block_dim - 1) / block_dim);
  axpy_kernel<T>
      <<<grid_dim, block_dim, 0, stream>>>(a, x, y, result, size);

  ThrowIfError(cudaGetLastError());
}

}

void gpu_axpy_f32(cudaStream_t stream, void **buffers, const char *opaque,
                    std::size_t opaque_len) {
  apply_axpy<float>(stream, buffers, opaque, opaque_len);
}

void gpu_axpy_f64(cudaStream_t stream, void **buffers, const char *opaque,
                    std::size_t opaque_len) {
  apply_axpy<double>(stream, buffers, opaque, opaque_len);
}

}