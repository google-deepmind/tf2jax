# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for tf2jax."""

from absl.testing import absltest
from absl.testing import parameterized

import jax
from jax.experimental import jax2tf
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from tf2jax._src import numpy_compat

_dtypes = [
    tf.bool, tf.uint8, tf.uint16, tf.uint32, tf.uint64, tf.int8, tf.int16,
    tf.int32, tf.int64, tf.bfloat16, tf.float16, tf.float32, tf.float64,
    tf.complex64, tf.complex128
]


class NumpyCompatTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("np", np, numpy_compat.tf_to_np_dtypes),
      ("jnp", jnp, numpy_compat.tf_to_jnp_dtypes),
  )
  def test_dtype_conversion(self, np_module, dtype_map):
    self.assertEqual(len(_dtypes), len(dtype_map))
    for src in _dtypes:
      dst = "bool_" if src.name == "bool" else src.name
      if src.name == "bfloat16":
        self.assertIs(dtype_map[src], jnp.bfloat16)
      else:
        self.assertIs(dtype_map[src], getattr(np_module, dst))

  def test_is_poly_dim(self):
    @jax.jit
    def jax_func(x):
      self.assertTrue(numpy_compat.is_poly_dim(x.shape[0]))
      self.assertEqual(x.shape[1], 4)
      return x + 3.14

    tf_func = jax2tf.convert(jax_func, polymorphic_shapes=["(b, _)"])
    tf_forward = tf.function(tf_func, autograph=False)
    tf_forward.get_concrete_function(tf.TensorSpec(shape=(None, 4)))


if __name__ == "__main__":
  absltest.main()
