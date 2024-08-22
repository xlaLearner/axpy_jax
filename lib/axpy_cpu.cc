/*
这是为了自己实现一个示例，即a * x + y的公式，以实现custom call
后续还需要利用xla_dump_to查看是否有custom call 的部分
*/

#include "axpy.h"
#include "pybind11_kernel_helpers.h"

using namespace axpy_jax;

namespace {

template <typename T>
void cpu_axpy(void *out_tuple, const void **in) {
    //分离出输入和输出
    const std::int64_t size = *reinterpret_cast<const std::int64_t *>(in[0]); 
    const T *a = reinterpret_cast<const T *>(in[1]);
    const T *x = reinterpret_cast<const T *>(in[2]);
    const T *y = reinterpret_cast<const T *>(in[3]);

    //由于只有一个输出，则只需要将out_tuple转换即可，而不需要像教程那样转换为2重指针
    T *result = reinterpret_cast<T *>(out_tuple);

    for (std::int64_t i = 0; i < size; ++i){
        compute_axpy<T>(a[i], x[i], y[i], result + i);
    }
    // compute_axpy(a, size, x, y, result);
}

pybind11::dict Registrations() {
    pybind11::dict dict;
    dict["cpu_axpy_float32"] = EncapsulateFunction(cpu_axpy<float>);
    dict["cpu_axpy_float64"] = EncapsulateFunction(cpu_axpy<double>);
    return dict;
}

//下面是将本文件代码绑定到python，第一个参数是python代码中的模块名称，m代表该模块，后面的m.def是将C++函数绑定到python的接口
//后面可以通过axpy.registrations进行调用里面的函数
PYBIND11_MODULE(axpy_cpu, m) { m.def("registrations", &Registrations); }

}