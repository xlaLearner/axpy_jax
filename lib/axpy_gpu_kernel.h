#ifndef _AXPY_JAX_AXPY_GPU_KERNELS_H_
#define _AXPY_JAX_AXPY_GPU_KERNELS_H_

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace axpy_jax {
struct AxpyDescriptor {
  std::int64_t size;
};

void gpu_axpy_f32(cudaStream_t stream, void** buffers, const char* opaque,
                    std::size_t opaque_len);
void gpu_axpy_f64(cudaStream_t stream, void** buffers, const char* opaque,
                    std::size_t opaque_len);

}  // namespace axpy_jax

#endif