/*
这里定义了暴露给XLA Custom Call 的python接口
*/

#include "axpy_gpu_kernel.h"
#include "pybind11_kernel_helpers.h"


using namespace axpy_jax;

namespace {

pybind11::dict Registrations() {
    pybind11::dict dict;
    dict["gpu_axpy_f32"] = EncapsulateFunction(gpu_axpy_f32);
    dict["gpu_axpy_f64"] = EncapsulateFunction(gpu_axpy_f64);
    return dict;
}

PYBIND11_MODULE(axpy_gpu, m) {
    m.def("registrations", &Registrations);
    m.def("build_axpy_descriptor",
            [](std::int64_t size) { return PackDescriptor(AxpyDescriptor{size}); });
}

}