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

"""Tests for quantization_util."""

from absl.testing import absltest
import jax
from jax import numpy as jnp
import numpy as np

from mediapipe.tasks.python.genai.converter import quantization_util


_dtype = lambda x: getattr(x, 'dtype', None) or np.asarray(x).dtype


class TestCase(absltest.TestCase):

  def assertAllClose(
      self, x, y, check_dtypes=True, rtol=1e-5, atol=1e-5, **kwargs
  ):
    """Wrapper for np.testing.assert_allclose()."""
    x = np.asarray(x)
    y = np.asarray(y)
    if check_dtypes:
      self.assertDtypesMatch(x, y)
    x = x.astype(np.float32) if x.dtype == jnp.bfloat16 else x
    y = y.astype(np.float32) if y.dtype == jnp.bfloat16 else y
    np.testing.assert_allclose(x, y, rtol=rtol, atol=atol, **kwargs)

  def assertDtypesMatch(self, x, y):
    self.assertEqual(
        jax.dtypes.canonicalize_dtype(_dtype(x)),
        jax.dtypes.canonicalize_dtype(_dtype(y)),
    )


class Quantize8BTest(TestCase):

  def test_quantize_symmetric(self):
    inputs = np.array([[1.2, 3.1, 5.5, 2.9], [0.2, -1.5, 3.3, 4.0]])
    qx, scale = quantization_util.quantize_tensor(inputs, axis=[1])

    self.assertAllClose(
        qx, np.array([[28, 72, 127, 67], [6, -48, 105, 127]], dtype=np.int8)
    )
    self.assertAllClose(
        scale, np.array([0.04330709, 0.03149606], dtype=np.float32)
    )

  def test_quantize_symmetric_with_dimension_size_one_unquantized(self):
    # inputs shape: (2, 1, 4), quantization axis 2.
    inputs = np.array([[[1.2, 3.1, 5.5, 2.9]], [[0.2, -1.5, 3.3, 4.0]]])
    qx, scale = quantization_util.quantize_tensor(inputs, axis=[2])

    self.assertAllClose(
        qx, np.array([[[28, 72, 127, 67]], [[6, -48, 105, 127]]], dtype=np.int8)
    )
    # expected scale shape: (2, 1)
    self.assertAllClose(
        scale, np.array([[0.04330709], [0.03149606]], dtype=np.float32)
    )

  def test_quantize_asymmetric(self):
    inputs = np.array([[1.2, 3.1, 5.5, 2.9], [0.2, -1.5, 3.3, 4.0]])
    qx, scale, zp = quantization_util.quantize_tensor(
        inputs, axis=[1], sym=False
    )

    self.assertAllClose(
        qx,
        np.array([[-128, -15, 127, -27], [-49, -128, 95, 127]], dtype=np.int8),
    )
    self.assertAllClose(scale, np.array([0.016863, 0.021569], dtype=np.float32))
    self.assertAllClose(zp, np.array([-3.358431, -1.260784], dtype=np.float32))


class Quantize8BFPTest(TestCase):

  def test_quantize_symmetric(self):
    inputs = np.array([[1.0, 2.0, 5.5, 2.9], [0.02, -0.01, 3.3, 4.0]])
    qx, scale = quantization_util.quantize_tensor(inputs, axis=[1], use_fp=True)

    self.assertAllClose(
        qx,
        np.array([[106, 114, 126, 119], [65, -71, 124, 126]], dtype=np.int8),
    )
    self.assertAllClose(
        scale, np.array([0.01227679, 0.00892857], dtype=np.float32)
    )

  def test_quantize_symmetric_with_dimension_size_one_unquantized(self):
    # inputs shape: (2, 1, 4), quantization axis 2.
    inputs = np.array([[[1.0, 2.0, 5.5, 2.9]], [[0.02, -0.01, 3.3, 4.0]]])
    qx, scale = quantization_util.quantize_tensor(inputs, axis=[2], use_fp=True)

    self.assertAllClose(
        qx,
        np.array(
            [[[106, 114, 126, 119]], [[65, -71, 124, 126]]], dtype=np.int8
        ),
    )
    # expected scale shape: (2, 1)
    self.assertAllClose(
        scale, np.array([[0.01227679], [0.00892857]], dtype=np.float32)
    )

  def test_quantize_asymmetric(self):
    inputs = np.array([[-1.0, -2.0, -2.01, 2.9], [0.02, -0.15, 3.3, 4.0]])
    qx, scale, zp = quantization_util.quantize_tensor(
        inputs, axis=[1], sym=False, use_fp=True
    )

    self.assertAllClose(
        qx,
        np.array([[-8, -2, -2, 126], [-3, -2, 121, 126]], dtype=np.int8),
    )
    self.assertAllClose(
        scale, np.array([0.00547991, 0.0046317], dtype=np.float32)
    )
    self.assertAllClose(
        zp, np.array([-0.4449999, -1.9250002], dtype=np.float32)
    )

  def test_quantize_add_scale_eps(self):
    inputs = np.array([[0.0, 0.0, 0.0, 0.0], [-4.0, -4.0, -4.0, -4.0]])
    _, scale, _ = quantization_util.quantize_tensor(
        inputs, axis=[1], sym=False, use_fp=True, add_scale_eps=True
    )
    self.assertAllClose(
        scale, np.array([np.finfo(np.float32).eps, np.finfo(np.float32).eps])
    )
    _, scale, _ = quantization_util.quantize_tensor(
        inputs, axis=[1], sym=False, use_fp=True, add_scale_eps=False
    )
    self.assertAllClose(scale, np.array([1.0, 1.0]))


class Quantize4BTest(TestCase):

  def test_quantize_symmetric(self):
    inputs = np.array([[1.2, 3.1, 5.5, 2.9], [0.2, -1.5, 3.3, 4.0]])
    qx, scale = quantization_util.quantize_tensor(
        inputs, axis=[1], number_bits=4
    )
    self.assertAllClose(
        qx, np.array([[2, 4, 7, 4], [0, -3, 6, 7]], dtype=np.int8)
    )
    self.assertAllClose(
        scale, np.array([0.78571427, 0.5714286], dtype=np.float32)
    )

  def test_quantize_symmetric_with_dimension_size_one_unquantized(self):
    # inputs shape: (2, 1, 4), quantization axis 2.
    inputs = np.array([[[1.2, 3.1, 5.5, 2.9]], [[0.2, -1.5, 3.3, 4.0]]])
    qx, scale = quantization_util.quantize_tensor(
        inputs, axis=[2], number_bits=4
    )

    self.assertAllClose(
        qx, np.array([[[2, 4, 7, 4]], [[0, -3, 6, 7]]], dtype=np.int8)
    )
    # expected scale shape: (2, 1)
    self.assertAllClose(
        scale, np.array([[0.78571427], [0.5714286]], dtype=np.float32)
    )

  def test_quantize_asymmetric(self):
    inputs = np.array([[1.2, 3.1, 5.5, 2.9], [0.2, -1.5, 3.3, 4.0]])
    qx, scale, zp = quantization_util.quantize_tensor(
        inputs, axis=[1], sym=False, number_bits=4
    )

    self.assertAllClose(
        qx,
        np.array([[-8, -1, 7, -2], [-3, -8, 5, 7]], dtype=np.int8),
    )
    self.assertAllClose(
        scale, np.array([0.2866667, 0.36666667], dtype=np.float32)
    )
    self.assertAllClose(
        zp, np.array([-3.4933336, -1.4333334], dtype=np.float32)
    )


class QuantizationUtilTest(TestCase):

  def test_update_to_uint4_sym(self):
    inputs = np.array([[1.2, 3.1, -5.5, 2.9], [0.2, -1.5, 3.3, 4.0]])
    qx, scale = quantization_util.quantize_tensor(
        inputs, axis=[1], sym=True, number_bits=4
    )
    dequant_from_int4 = qx * np.expand_dims(scale, -1)
    qx_n, scale_n, zp_n = quantization_util.update_to_uint4(qx, scale)
    self.assertEmpty(zp_n.shape)  # A scalar numpy array.
    dequant_from_uint4 = np.expand_dims(scale_n, -1) * (qx_n - zp_n)
    np.testing.assert_allclose(dequant_from_int4, dequant_from_uint4)

  def test_update_to_uint4_sym_combined(self):
    inputs = np.array(
        [[-1.2, 3.5, -6.2, 1.7], [1.2, 3.1, -5.5, 2.9], [0.2, -1.5, 3.3, 4.0]]
    )
    qx, scale = quantization_util.quantize_tensor(
        inputs, axis=[1], sym=True, number_bits=4
    )
    dequant_from_int4 = qx * np.expand_dims(scale, -1)
    qx_n, scale_n, zp_n = quantization_util.update_to_uint4(qx, scale)
    self.assertEqual(zp_n.shape[0], 3)
    dequant_from_uint4 = np.expand_dims(scale_n, -1) * (
        qx_n - np.expand_dims(zp_n, -1)
    )
    np.testing.assert_allclose(dequant_from_int4, dequant_from_uint4)

  def test_update_to_uint4_asym(self):
    inputs = np.array([[1.0, 8.0, -3.0, 2.0], [-3.0, 2.0, 1.0, 8.0]])
    qx, scale, zp = quantization_util.quantize_tensor(
        inputs, axis=[1], sym=False, number_bits=4
    )
    qx_n, scale_n, zp_n = quantization_util.update_to_uint4(qx, scale, zp)
    expected_dequant = np.array([
        [0.0, 7.333333, -3.666667, 1.466667],
        [-3.666667, 1.466667, 0.0, 7.333333],
    ])
    dequant_from_uint4 = np.expand_dims(scale_n, -1) * (qx_n - zp_n)
    np.testing.assert_allclose(dequant_from_uint4, expected_dequant, rtol=1e-05)

  def test_update_to_uint4_asym_combined(self):
    inputs = np.array(
        [[1.0, 8.0, -3.0, 2.0], [-3.0, 2.0, 1.0, 8.0], [2.0, 1.0, 8.0, -3.0]]
    )
    qx, scale, zp = quantization_util.quantize_tensor(
        inputs, axis=[1], sym=False, number_bits=4
    )
    qx_n, scale_n, zp_n = quantization_util.update_to_uint4(qx, scale, zp)
    self.assertEqual(zp_n.shape[0], 3)
    expected_dequant = np.array([
        [0.0, 7.333333, -3.666667, 1.466667],
        [-3.666667, 1.466667, 0.0, 7.333333],
        [1.466667, 0.0, 7.333333, -3.666667],
    ])
    dequant_from_uint4 = np.expand_dims(scale_n, -1) * (
        qx_n - np.expand_dims(zp_n, -1)
    )
    np.testing.assert_allclose(dequant_from_uint4, expected_dequant, rtol=1e-05)


class QuantizeMseTest(TestCase):

  def test_mse_reduce_precision_4bit(self):
    inputs = np.array([[-1.0, 1.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float32)
    qx, scale = quantization_util.quantize_tensor(
        inputs, axis=[1], number_bits=4, use_mse_quant=True
    )

    # For 4 bits, maxq = 15, zero_mse_quant = (15 - 1) / 2 = 7.
    # The scale multiplier is 0.37755.

    # For inputs row 1 ([-1.0, 1.0, 0.0]):
    # rms = sqrt(mean((-1)^2 + 1^2 + 0^2)) = sqrt(2/3) approx 0.8164966
    # scale = 0.37755 * 0.8164966 = 0.3079979
    # t_shifted(-1.0) = -1.0 / 0.3079979 + 7 = -3.2461 + 7 = 3.7539
    #   -> round 4 -> qvar = -3
    # t_shifted(1.0) = 1.0 / 0.3079979 + 7 = 3.2461 + 7 = 10.2461
    #   -> round 10 -> qvar = 3
    # t_shifted(0.0) = 0.0 / 0.3079979 + 7 = 7.0 -> round 7 -> qvar = 0

    # For inputs row 2 ([1.0, 2.0, 3.0]):
    # rms = sqrt(mean(1^2 + 2^2 + 3^2)) = sqrt(14/3) approx 2.1602469
    # scale = 0.37755 * 2.1602469 = 0.8159988
    # t_shifted(1.0) = 1.0 / 0.8159988 + 7 = 1.2255 + 7 = 8.2255
    #   -> round 8 -> qvar = 1
    # t_shifted(2.0) = 2.0 / 0.8159988 + 7 = 2.4509 + 7 = 9.4509
    #   -> round 9 -> qvar = 2
    # t_shifted(3.0) = 3.0 / 0.8159988 + 7 = 3.6764 + 7 = 10.6764
    #   -> round 11 -> qvar = 4
    self.assertAllClose(qx, np.array([[-3, 3, 0], [1, 2, 4]], dtype=np.int8))
    self.assertAllClose(scale, np.array([0.308268, 0.815601], dtype=np.float32))


if __name__ == '__main__':
  absltest.main()
