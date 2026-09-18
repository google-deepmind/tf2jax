# Copyright 2022 DeepMind Technologies Limited. All Rights Reserved.
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
"""Test utils."""

import contextlib

from absl import logging
from absl.testing import parameterized

import jax
import tensorflow as tf


def parse_version(version: str):
  return tuple(int(x.split("-")[0]) for x in version.split("."))


class TestCase(parameterized.TestCase, tf.test.TestCase):
  """Base class for all tests."""

  def setUp(self):
    super().setUp()
    # Ensure that all TF ops are created on the proper device (TPU, GPU or CPU)
    jax_default_device = jax.default_backend().upper()
    tf_logical_devices = tf.config.list_logical_devices(jax_default_device)
    self._tf_on_cpu_fallback = False
    if tf_logical_devices:
      tf_default_device = tf_logical_devices[0]
      self.assertEqual(jax_default_device, tf_default_device.device_type)
    else:
      tf_default_device = tf.config.list_logical_devices("CPU")[0]
      self._tf_on_cpu_fallback = True
    logging.info(
        "Running JAX on %s and TF on %s.", jax_default_device, tf_default_device
    )

    with contextlib.ExitStack() as stack:
      stack.enter_context(tf.device(tf_default_device))
      if self._tf_on_cpu_fallback and jax_default_device == "TPU":
        stack.enter_context(jax.default_matmul_precision("float32"))
      self.addCleanup(stack.pop_all().close)

  def assertAllClose(self, a, b, rtol=1e-6, atol=1e-6, msg=None):  # pylint: disable=invalid-name
    if getattr(self, "_tf_on_cpu_fallback", False):
      rtol = max(rtol, 1e-4)
      atol = max(atol, 1e-4)
    super().assertAllClose(a, b, rtol=rtol, atol=atol, msg=msg)
