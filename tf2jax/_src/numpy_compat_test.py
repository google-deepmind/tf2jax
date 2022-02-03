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

import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from tf2jax._src import numpy_compat

_dtypes = [
    "bool", "uint8", "uint16", "uint32", "uint64", "int8", "int16", "int32",
    "int64", "float16", "float32", "float64", "complex64", "complex128"
]


class NumpyCompatTest(parameterized.TestCase):

  def test_numpy_dtype_conversion(self):
    self.assertEqual(len(_dtypes), len(numpy_compat.tf_to_np_dtypes))
    for src in _dtypes:
      dst = "bool_" if src == "bool" else src
      self.assertEqual(numpy_compat.tf_to_np_dtypes[getattr(tf, src)],
                       getattr(np, dst))

  def test_jax_dtype_conversion(self):
    all_dtypes = _dtypes + ["bfloat16"]
    self.assertEqual(len(all_dtypes), len(numpy_compat.tf_to_jnp_dtypes))
    for src in all_dtypes:
      dst = "bool_" if src == "bool" else src
      self.assertEqual(numpy_compat.tf_to_jnp_dtypes[getattr(tf, src)],
                       getattr(jnp, dst))


if __name__ == "__main__":
  absltest.main()
