
__all__ = ["axpy"]

from functools import partial

import numpy as np
from jax import core, dtypes, lax
from jax import numpy as jnp
from jax.core import ShapedArray
from jax.interpreters import ad, batching, mlir, xla
from jax._src.lib.mlir.dialects import hlo
from jax.lib import xla_client
from jaxlib.hlo_helpers import custom_call

# 注册XLA中CPU代码
from . import axpy_cpu

for _name, _value in axpy_cpu.registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform="cpu")

# 检测是否有gpu版本，一并注册
# TODO：TPU怎么实现？使用JAX自己的pallas
try:
    from . import axpy_gpu
except ImportError:
    axpy_gpu = None
else:
    for _name, _value in axpy_gpu.reregistrations().items():
        xla_client.register_custom_call_target(_name, _value, platform="gpu")

# 注册后，写一个JAX原语

def axpy(a, x, y):
    x, y = jnp.broadcast_arrays(x, y)
    a, x = jnp.broadcast_arrays(a, x)
    return _axpy_prim.bind(a, x, y)

def axpy_impl(a, x, y):
    return jnp.add(jnp.multiply(a, x), y)

# 要使得该函数能被jit，需要Abstract evaluation rules,实际上就是一些检查工作
def axpy_abstract_eval(sa, sx, sy):
    shape = sx.shape
    dtype = dtypes.canonicalize_dtype(sx.dtype)
    assert shape == sy.shape
    assert shape == sa.shape
    assert dtype == dtypes.canonicalize_dtype(sy.dtype)
    assert dtype == dtypes.canonicalize_dtype(sa.dtype)
    
    # 下面实际上是返回了输入的shape和dtype
    return (ShapedArray(shape, dtype), ShapedArray(shape, dtype), ShapedArray(shape, dtype))

# 还需要确定lowering操作，我的理解是业务代码当中的一个函数对应了后端中如何表示，这里的lowering会交给XLA/告诉XLA怎么转换当前函数
def axpy_lowering(ca, cx, cy):
    """The compilation to XLA of the primitive.

    Given an mlir.ir.Value for each argument, return the mlir.ir.Values for
    the results of the function.

    Does not need to be a JAX-traceable function.
    """
    # 这里仅仅是乘加运算，这些运算在XLA中已经被定义，所以可以使用hlo当中对应的Op
    # 如果更加复杂的计算被定义，无法使用XLA中原有的Op，则要类似于kepler，需要接上custom call接口，而不是使用hlo当中的Op
    return [hlo.AddOp(hlo.MulOp(ca, cx), cy).result]

# 前向微分，定义自动微分方式
# r = ax + y -> dr = a * dx + dy
def axpy_jvp(args, tangents):
    a, x, y = args
    at, xt, yt = tangents

    primal_output = _axpy_prim(a, x, y)

    def zero_tan(tan):
        return lax.zeros_like_array(x) if type(tan) is ad.Zero else tan
    tangent_output = _axpy_prim(zero_tan(at), x, _axpy_prim(a, zero_tan(xt), zero_tan(yt)))
    return (primal_output, tangent_output)

# 暂时用不到vmap，暂时不实现batching



_axpy_prim = core.Primitive("axpy")
_axpy_prim.multiple_results = False
_axpy_prim.def_impl(partial(xla.apply_primitive), axpy_impl)
_axpy_prim.def_abstract_eval(axpy_abstract_eval)

# 告诉XLA转换规则
for platform in ["cpu", "gpu"]:
    mlir.register_lowering(
        _axpy_prim,
        partial(axpy_lowering, platform=platform),
        platform=platform)
    
ad.primitive_jvps[_axpy_prim] = axpy_jvp
