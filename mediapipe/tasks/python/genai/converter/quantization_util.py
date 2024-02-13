# Copyright 2024 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for quantizing tensors.

Note that this is a reduced fork version of the praxis libraries to provide a
self-contained library for packaging.
"""

from typing import Any, List, Optional, Sequence, Tuple, Union

import jax
from jax import lax
from jax import numpy as jnp
import numpy as np


JTensor = jax.Array
_UINT4_ZP = 8  # Default zero point for unsigned 4-bit.


def _get_scan_range() -> np.ndarray:
  # Produce candidate scan values.
  return np.linspace(1.0, 0.5, num=11)


def _get_mean_error(bound, t, min_value, max_value, p_value):
  scale = bound / max_value
  candidate = jnp.divide(t, scale)
  candidate = jnp.clip(jnp.round(candidate), min_value, max_value)
  candidate = jnp.multiply(candidate, scale)
  pmean_error = jnp.mean(jnp.abs(jnp.subtract(candidate, t)) ** p_value)
  return pmean_error


def _get_best_bound_per_tensor(
    t: JTensor,
    bound: JTensor,
    min_value: float,
    max_value: float,
    p_value: float = 1.0,
) -> JTensor:
  """Scan around [0.5, 1] * hard max value to get bound value for whole tensor.

  This does a scan to get bound value(s) that minimize mean absolute error (MAE)
  between original tensor 't' and quantized tensor. It's (almost) equivalent to
  maximizing entropy.

  Args:
    t: The input float tensor.
    bound: The hard max value for tensor 't'. It has the same length as shape.
    min_value: Minimal value for the quantization bound.
    max_value: Maximal value for the quantization bound.
    p_value: Exponent of the p-mean error metric. Default to 1.0 which is MAE.

  Returns:
    The best bound values for 't', that minimize p-mean error.
  """

  def _quant(scaling_factors):
    return _get_mean_error(
        bound * scaling_factors, t, min_value, max_value, p_value
    )

  scaling_factors = _get_scan_range()
  diffs = jax.vmap(_quant)(scaling_factors)
  best_scaling = scaling_factors[jnp.argmin(diffs)].astype(bound.dtype)
  return bound * best_scaling


def _quantrow(
    vec: JTensor,
    bound: JTensor,
    min_value: float,
    max_value: float,
    p_value: float,
    factors: np.ndarray,
) -> JTensor:
  """Get best rescaling factor from a list of factors applied a channel.

  Args:
    vec: The vector in a channel.
    bound: The hard bound (max(abs(vec))) of the vector.
    min_value: The target min value.
    max_value: The target max value.
    p_value: Exponent of the p-mean error metric.
    factors: The values to be applied on top of bound.

  Returns:
    adjusted bound value out of the list of factors applied to bound.
  """

  def _quant(bounds):
    return _get_mean_error(bounds, vec, min_value, max_value, p_value)

  diffs = jax.vmap(_quant)(bound * factors)
  best_scaling = factors[jnp.argmin(diffs)]
  return bound * best_scaling


def _get_best_bound_per_channel(
    t: JTensor,
    bound: JTensor,
    min_value: float,
    max_value: float,
    p_value: float = 1.0,
) -> JTensor:
  """Scan around [0.5, 1] * hard max value to get bound value for each channel.

  This does a scan to get bound value(s) that minimize mean absolute error (MAE)
  between original tensor 't' and quantized tensor. It's (almost) equivalent to
  maximizing entropy.

  Args:
    t: The input float tensor.
    bound: The hard max value for tensor 't'. It has the same length as shape.
    min_value: Minimal value for the quantization bound.
    max_value: Maximal value for the quantization bound.
    p_value: Exponent of the p-mean error metric. Default to 1.0 which is MAE.

  Returns:
    The best bound values for 't', that minimize p-mean error.
  """
  assert len(t.shape) == 2
  assert len(bound.shape) == 2
  assert t.shape[1] == bound.shape[1]
  assert bound.shape[0] == 1
  scans = _get_scan_range()

  def _quant(tensor, bound, min_value, max_value, p_value, factors):
    ret = np.zeros(bound.shape)
    for i in range(len(tensor)):
      best = _quantrow(
          tensor[i], bound[i], min_value, max_value, p_value, factors
      )
      ret[i] = best
    return ret

  t = t.transpose()
  t_split = list(t)
  res = _quant(t_split, bound[0, :], min_value, max_value, p_value, scans)
  res = res.reshape(bound.shape)
  return res


def get_best_bound(
    t: JTensor,
    bound: JTensor,
    min_value: float,
    max_value: float,
    p_value: float = 1.0,
    per_channel: bool = False,
) -> JTensor:
  """Scan mutliple factors on max value to get best bound value.

  This does a scan to get bound value(s) that minimize mean absolute error (MAE)
  between original tensor 't' and quantized tensor. It's (almost) equivalent to
  maximizing entropy.

  Args:
    t: The input float tensor.
    bound: The hard max value for tensor 't'. It has the same length as shape.
    min_value: Minimal value for the quantization bound.
    max_value: Maximal value for the quantization bound.
    p_value: Exponent of the p-mean error metric. Default to 1.0 which is MAE.
    per_channel: if get best bound for entire tensor or per channel.

  Returns:
    The best bound values for 't', that minimize p-mean error.
  """
  if per_channel:
    return _get_best_bound_per_channel(t, bound, min_value, max_value, p_value)
  else:
    return _get_best_bound_per_tensor(t, bound, min_value, max_value, p_value)


def get_min_max(
    bits: int = 8,
    unsigned: bool = False,
    use_fp: bool = False,
) -> Tuple[float, float]:
  """Gets the min/max range for a given number of bits.

  Args:
    bits: Target number of bits for quantization.
    unsigned: If True compute min and max for unsigned number, else for signed.
    use_fp: in floating point.

  Returns:
    min/max values for the provide number of bits.
  """
  if use_fp:
    # TODO: support other fp types.
    return -448.0, 448.0
  # Calculation instead of jax.iinfo is used to support bits beside 4 and 8.
  if unsigned:
    # For unsigned 8 bits precision it is [0, 255]
    return 0, 2**bits - 1
  else:
    # For signed 8 bits precision it is [-128, 127]
    return -1 * 2 ** (bits - 1), 2 ** (bits - 1) - 1


def pass_through(x: JTensor, fn: Any) -> JTensor:
  # Create an exactly-zero expression with Sterbenz lemma that has an
  # exactly-one gradient.
  return x - jax.lax.stop_gradient(x) + jax.lax.stop_gradient(fn(x))


def reduce_precision(
    t: JTensor,
    contract_dims: Optional[Sequence[int]],
    need_gradient: bool = False,
    bits: int = 8,
    optimization_on_bound: bool = False,
    p_value: float = 1.0,
    percentile: float = 1.0,
    use_symmetric: bool = True,
    use_fp: bool = False,
    add_scale_eps: bool = False,
    per_channel: bool = False,
    random_rounding: bool = False,
    key: Optional[jax.Array] = None,
) -> Tuple[JTensor, JTensor, Optional[JTensor]]:
  """Reduce the precision of a tensor.

  Generic for all tensors.

  Args:
    t: Input tensor.
    contract_dims: Speficies contracting dimesnions of the input tensor.
    need_gradient: If gradient is needed out of this function.
    bits: Target number of bits.
    optimization_on_bound: If MAE bound optimizer is used.
    p_value: Exponent of the p-mean error metric. Default to 1.0 which is MAE.
    percentile: Percentile Factor to apply on the min/max range. Setting this to
      other than 1.0 disables optimization_on_bound.
    use_symmetric: If the input tensor is quantized symmetrically.
    use_fp: Use floating point.
    add_scale_eps: Add eps value or replace zero value by 1 to avoid division by
      zero.
    per_channel: use per-channel clipping optimization.
    random_rounding: round with uniform random.
    key: rng key for rounding.

  Returns:
    A tuple of quantized tensor, quantization scale
      and quantization zero point (optional).
  """
  min_value, max_value = get_min_max(bits, use_fp=use_fp)

  if use_symmetric:
    bound = jnp.max(jnp.abs(t), axis=contract_dims, keepdims=True)
    scale_bound = max_value
  else:
    t_max = jnp.max(t, axis=contract_dims, keepdims=True)
    t_min = jnp.min(t, axis=contract_dims, keepdims=True)
    bound = t_max - t_min
    scale_bound = max_value - min_value

  if percentile < 1.0:
    bound = jnp.multiply(bound, percentile)
  elif optimization_on_bound:
    bound = get_best_bound(
        t, bound, min_value, max_value, p_value, per_channel=per_channel
    )

  scale = bound / scale_bound

  if add_scale_eps:
    # Add epsilon to avoid divide-by-zero.
    scale = scale + jnp.finfo(t.dtype).eps
  else:
    scale = jnp.where(scale == 0.0, 1.0, scale)

  if use_symmetric:
    zp = None
    t = jnp.divide(t, scale)
  else:
    zp = min_value - t_min / scale
    t = jnp.divide(t, scale) + zp
    zp = jnp.multiply(scale, zp)

  if use_fp:
    # No need to round.
    t = jnp.clip(t, min_value, max_value).astype(jnp.float8_e4m3fn)
    # TODO: refactor to remove this logic.
    t = jax.lax.bitcast_convert_type(t, new_dtype=jnp.int8)
  else:
    if need_gradient:
      t = pass_through(t, jnp.round)
      t = jnp.clip(t, min_value, max_value)
    else:
      if random_rounding:
        t = t + jax.random.uniform(
            key=key, shape=t.shape, minval=-0.5, maxval=0.5
        )
      t = jnp.round(t)
      container_dtype = (
          jnp.int8 if bits <= 8 else jnp.int16 if bits <= 16 else jnp.int32
      )
      t = jnp.clip(t, min_value, max_value).astype(container_dtype)

  return t, scale, zp


def quantize_tensor(
    var: np.ndarray,
    axis: List[int],
    factor: float = 1.0,
    sym: bool = True,
    number_bits: int = 8,
    use_fp: bool = False,
    add_scale_eps: bool = False,
    optimization_on_bound: bool = False,
    p_value: float = 1.0,
    per_channel: bool = False,
    block_size: int = 0,
) -> Union[
    Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]
]:
  """Quantize a tensor.

  Args:
    var: The variable to be quantized.
    axis: The axis along which variable will be quantized.
    factor: The clipping factor.
    sym: Symmetric or asymmetric quantize the variable.
    number_bits: Number of bits for quantized value.
    use_fp: do fp with number of bits (i.e. fp8)
    add_scale_eps: add epsilon to scale to avoid division by zero, else it will
      replace zero scale by 1.
    optimization_on_bound: If p-mean bound optimizer is used.
    p_value: Exponent of the p-mean error metric. Default to 1.0 which is MAE.
    per_channel: use per-channel clipping optimization.
    block_size:  block size for sub-channel quantization. Defaults to 0, which
      means off.

  Returns:
    Quantized tensors, along with scales and zero point.
  """
  # TODO: support jnp.float8_e5m2
  assert number_bits == 8 or number_bits == 4
  jnp_var = jnp.asarray(var)
  # When using sub-channel, the contracting dim is split into a sub-channel
  # dim followed by the block dim. Therefore the contracting dim
  # (quantize_axis) should increment by one, and the corresponding pack_dim
  # should also increment by one.
  if block_size > 0:
    shape = list(jnp_var.shape)
    assert len(axis) == 1, 'Only support 1D sub-channel quantization'
    sub_channels, rem = divmod(shape[axis[0]], block_size)
    assert rem == 0
    shape.insert(axis[0], sub_channels)
    axis[0] += 1
    shape[axis[0]] = block_size
    jnp_var = jnp.reshape(jnp_var, shape)

  qvar, scale, zp = reduce_precision(
      jnp_var,
      contract_dims=axis,
      need_gradient=False,
      bits=number_bits,
      optimization_on_bound=optimization_on_bound,
      percentile=factor,
      use_symmetric=sym,
      use_fp=use_fp,
      add_scale_eps=add_scale_eps,
      p_value=p_value,
      per_channel=per_channel,
  )
  if sym:
    return np.array(qvar), np.array(jnp.squeeze(scale, axis=axis))  # pytype: disable=wrong-arg-types  # jnp-type
  else:
    return (
        np.array(qvar),
        # CAVEAT: the following squeezes should squeeze along the quantization
        # axis only.
        np.array(jnp.squeeze(scale)),
        np.array(jnp.squeeze(zp)),
    )


def pack_4bit(
    x: np.ndarray, pack_dim: int, packed_dtype: jnp.dtype = jnp.int32
) -> np.ndarray:
  """Pack int8 or uint8 tensor where its values are actually int4 or uint4, to int32 or int8 nibble format along pack_dim.

  Args:
    x: Original int8 or uint8 tensor to pack.
    pack_dim: Dimension to pack along. x.shape[pack_dim] must be divisible by 8,
      when packed_dtype is int32 and divisible by 2 when target_type is int8.
      Also pack_dim must be < x.ndim - 1.
    packed_dtype: Target type to pack to, int32 or int8.

  Returns:
    int32 or int8 packed tensor where the pack_dim size is dividened by 8
    from the original tensor x.
  """
  x = jnp.asarray(x)
  if packed_dtype == jnp.int8 and x.dtype == jnp.uint8:
    # It doesn't make sense to pack uint8 numbers into int4 as we'll
    # the range overlap between uint8 and int4 is [0..7].
    raise ValueError(
        'only int8 input dtype is supported when packing into int8. '
        f'Given {x.dtype}'
    )

  if x.dtype != jnp.int8 and x.dtype != jnp.uint8:
    raise ValueError(
        f'input dtype must be either int8 or uint8. Given {x.dtype}'
    )
  if pack_dim >= x.ndim - 1:
    raise ValueError(
        f'pack_dim must be < input ndim - 1. input shape {x.shape} and pack_dim'
        f' {pack_dim}'
    )
  if packed_dtype != jnp.int32 and packed_dtype != jnp.int8:
    raise ValueError(
        f'packed_dtype must be either int32 or int8. Given {packed_dtype}'
    )
  if packed_dtype == jnp.int32 and x.shape[pack_dim] % 8 != 0:
    raise ValueError(
        'input shape[pack_dim] must be divisible by 8 when target_type '
        f'is int32. Given shape {x.shape}'
    )
  if packed_dtype == jnp.int8 and x.shape[pack_dim] % 2 != 0:
    raise ValueError(
        'input shape[pack_dim] must be divisible by 2 when target_type '
        f'is int8. Given shape {x.shape}'
    )

  int4s_per_packed_type = 8 if packed_dtype == jnp.int32 else 2

  rep_shape = list(x.shape)
  rep_shape.insert(pack_dim + 1, int4s_per_packed_type)
  rep_shape[pack_dim] //= int4s_per_packed_type

  shifts = lax.broadcasted_iota(packed_dtype, rep_shape, pack_dim + 1)
  shifts <<= 2

  # Promote x to packed_dtype
  x = x & jnp.array(0x0F, packed_dtype)
  x = lax.reshape(x, rep_shape)
  x = x << shifts
  x = lax.reduce(x, jnp.array(0x0, packed_dtype), lax.add, [pack_dim + 1])
  return np.asarray(x)


def update_to_uint4(
    qx: np.ndarray, scale: np.ndarray, zp: Optional[np.ndarray] = None
):
  """Updates the quantized weights from int4 to uint4.

  This is a conversion function designed for XNNPack as it expects the 4-bit
  quantized weight to be represented differently from the original Pax setting.
  Specifically, the differences are:
    1) The dynamic range of weight values: int4 (Pax) vs. uint4 (XNNPack).
    2) The dynamic range of zero-point: float (Pax) vs. uint4 (XNNPack).
    3) The number of zero-point: per-channel (Pax) vs. per-tensor (XNNPack).

  Args:
    qx: np.array of shape [..., channel], which is the quantized weight values
      from Pax in the shape of. The values are in the dynamic range of int4 but
      are hosted as int8 type. Note that if the first dimension is 3, it means
      the qkv matrices are concatenated together and should be treated
      differently.
    scale: np.array of shape [1(3), channel] as np.float type, which are the
      scaling factors for dequantization per channel.
    zp: (optional) np.array of shape [1 (or 3), channel] as np.float type, which
      are the zero points for dequantization per channel.

  Returns:
    A tuple (qx, scale, zp):
      qx: The updated np.array of shape [..., channel] as np.int8 type with
        updated dynamic range as uint4 (with 8 as the default zero points).
      scale: Same as the input scale.
      zp: (optional) np.array of shape [1 (or 3)] as np.int8 type with the
        updated zero point values in the dynamic range as uint4.
  """
  if qx.dtype != np.int8 or ('float' not in str(scale.dtype)):
    raise ValueError(
        'Unexpected dtype qx:' + str(qx.dtype) + ' scale:' + str(scale.dtype)
    )

  scale = scale.astype(np.float32)

  def get_new_zp(old_zp):
    new_zp = old_zp / (scale + np.finfo(np.float32).eps)
    per_tensor_zp = np.mean(new_zp)
    per_tensor_zp = per_tensor_zp.astype(np.int8) + _UINT4_ZP
    return per_tensor_zp

  if zp is not None:
    if qx.shape[0] == 3:
      per_tensor_zp = np.stack([get_new_zp(szp) for szp in zp], axis=0)
    else:
      per_tensor_zp = get_new_zp(zp)
  else:
    per_tensor_zp = (
        _UINT4_ZP * np.ones(shape=(3)) if qx.shape[0] == 3 else _UINT4_ZP
    )

  qx = qx + _UINT4_ZP
  return qx, scale, np.array(per_tensor_zp, dtype=np.int32)
