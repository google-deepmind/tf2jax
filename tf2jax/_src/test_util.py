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
    tf_default_device = tf_logical_devices[0]
    logging.info("Running tf2jax converted code on %s.", tf_default_device)
    self.assertEqual(jax_default_device, tf_default_device.device_type)

    with contextlib.ExitStack() as stack:
      stack.enter_context(tf.device(tf_default_device))
      self.addCleanup(stack.pop_all().close)
