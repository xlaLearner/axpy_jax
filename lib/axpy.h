/*
cpu后端对axpy的实际实现，模仿kepler.h
*/

#ifndef _AXPY_JAX_AXPY_H_
#define _AXPY_JAX_AXPY_H_

#include <cmath>

namespace axpy_jax {


#ifdef __CUDACC__
#define SJH_JAX_INLINE_OR_DEVICE __host__ __device__
#else
#define SJH_JAX_INLINE_OR_DEVICE inline

#endif

// axpy需要3个输入,1个输出
template <typename T>
SJH_JAX_INLINE_OR_DEVICE void compute_axpy(T* a, T* x, T* y, T* result) {
  result = (*a) * (*x) + *y
}

}  // namespace axpy_jax

#endif