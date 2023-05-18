# Copyright 2023 DeepMind Technologies Limited. All Rights Reserved.
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
"""Tests for JAX -> TF -> JAX with partitioning."""

from absl.testing import absltest
from absl.testing import parameterized

import chex
import haiku as hk
import jax
from jax.experimental import jax2tf
import jax.numpy as jnp

import numpy as np
import tensorflow as tf
from tf2jax._src import test_util
from tf2jax._src import tf2jax
import tree


def _net(x):
  y = hk.Conv2D(32, kernel_shape=3, name='A1')(x)
  y = hk.Conv2D(32, kernel_shape=3, name='A2')(y)
  return y


def _get_param_pspecs():
  return {
      'A1': {
          'b': jax.sharding.PartitionSpec('model'),
          'w': jax.sharding.PartitionSpec(None, None, None, 'model'),
      },
      'A2': {
          'b': jax.sharding.PartitionSpec(None),
          'w': jax.sharding.PartitionSpec(None, None, 'model', None),
      },
  }


class ShardingTest(test_util.TestCase):

  @parameterized.named_parameters(
      chex.params_product(
          (('enable_xla', True), ('disable_xla', False)),
          (('native_serialization', True), ('graph_serialization', False)),
          named=True,
      )
  )
  def test_sharding(self, enable_xla, native_serialization):
    if jax.default_backend().upper() != 'TPU':
      self.skipTest('Only run sharding tests on TPU.')

    if not enable_xla and native_serialization:
      self.skipTest(
          'native_serializaton is only supported with enable_xla=True.'
      )

    # Set up network and inputs.
    transformed = hk.without_apply_rng(hk.transform(_net))
    rng = jax.random.PRNGKey(42)
    images = jax.random.normal(rng, [2, 16, 16, 3])
    grad_tols = dict(rtol=1e-4)

    # Init params.
    params = jax.jit(transformed.init)(rng, images)

    # Partitioned to 8 devices.
    assert jax.device_count() == 8, jax.device_count()
    mesh = jax.sharding.Mesh(
        np.array(jax.devices()).reshape((2, 4)), ('data', 'model'))
    params_pspecs = _get_param_pspecs()
    def to_xla_sharding(pspecs):
      return jax.tree_map(
          lambda x: jax.sharding.NamedSharding(mesh, x), pspecs)

    partitioned_apply = jax.jit(
        transformed.apply,
        in_shardings=to_xla_sharding(
            (params_pspecs, jax.sharding.PartitionSpec('data'))
        ),
    )
    self.assertAllClose(
        jax.jit(transformed.apply)(params, images),
        partitioned_apply(params, images),
    )

    # Check gradients.
    @jax.grad
    def unpartitioned_grad(params, xs):
      return jnp.sum(jax.jit(transformed.apply)(params, xs))

    @jax.grad
    def partitioned_grad(params, xs):
      return jnp.sum(partitioned_apply(params, xs))

    self.assertAllClose(
        unpartitioned_grad(params, images),
        partitioned_grad(params, images),
        **grad_tols,
    )

    # Convert to TF and save.
    @tf.function(autograph=False, jit_compile=True)
    def tf_fn(params, inputs):
      return jax2tf.convert(
          partitioned_apply,
          enable_xla=enable_xla,
          native_serialization=native_serialization,
      )(params, inputs)

    tf_fn(params, images)
    module = tf.Module()
    module.f = tf_fn
    export_dir = self.create_tempdir().full_path
    tf.saved_model.save(module, export_dir)

    # Load and tf2jax
    reloaded = tf.saved_model.load(export_dir)
    jax_fn = tf2jax.convert_functional(
        tf.function(reloaded.f, autograph=False),
        params=tree.map_structure(np.zeros_like, params),
        inputs=np.zeros_like(images),
    )
    self.assertAllClose(
        transformed.apply(params, images), jax_fn(params, images)
    )
    self.assertAllClose(
        jax.jit(transformed.apply)(params, images),
        jax.jit(jax_fn)(params, images),
    )

    # Check gradients.
    @jax.grad
    def reloaded_grad(params, xs):
      return jnp.sum(jax.jit(jax_fn)(params, xs))
    self.assertAllClose(
        jax.jit(unpartitioned_grad)(params, images),
        jax.jit(reloaded_grad)(params, images),
        **grad_tols,
    )

    # Check shardings.
    unpartitioned_output = jax.jit(transformed.apply)(params, images)
    self.assertLen(unpartitioned_output.devices(), 1)
    partitioned_output = partitioned_apply(params, images)
    self.assertLen(partitioned_output.devices(), 8)
    reloaded_output = jax.jit(jax_fn)(params, images)
    self.assertLen(reloaded_output.devices(), 8)
    self.assertTrue(
        partitioned_output.sharding.is_equivalent_to(
            reloaded_output.sharding, 4
        )
    )


if __name__ == '__main__':
  absltest.main()
