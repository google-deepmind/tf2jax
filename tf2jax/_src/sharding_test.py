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
from flax import linen as nn
import jax
from jax.experimental import jax2tf
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from tf2jax._src import test_util
from tf2jax._src import tf2jax
import tree


class FlaxNet(nn.Module):

  @nn.compact
  def __call__(self, x):
    y = nn.Conv(features=32, kernel_size=(3, 3), name='A1')(x)
    y = nn.Conv(features=32, kernel_size=(3, 3), name='A2')(y)
    return y


def _get_param_pspecs():
  return {
      'A1': {
          'bias': jax.sharding.PartitionSpec('model'),
          'kernel': jax.sharding.PartitionSpec(None, None, None, 'model'),
      },
      'A2': {
          'bias': jax.sharding.PartitionSpec(None),
          'kernel': jax.sharding.PartitionSpec(None, None, 'model', None),
      },
  }


class ShardingTest(test_util.TestCase):

  def test_sharding(self):
    if jax.default_backend().upper() != 'TPU':
      self.skipTest('Only run sharding tests on TPU.')

    # Set up network and inputs.
    transformed = FlaxNet()
    rng = jax.random.key(42)
    images = jax.random.normal(rng, [2, 16, 16, 3])
    grad_tols = dict(rtol=1e-4)

    # Init params.
    variables = jax.jit(transformed.init)(rng, images)
    params = variables['params']  # Extract the parameters
    transformed_apply = lambda p, x: transformed.apply({'params': p}, x)

    # Partitioned to 8 devices.
    assert jax.device_count() == 8, jax.device_count()
    mesh = jax.sharding.Mesh(
        np.array(jax.devices()).reshape((2, 4)), ('data', 'model')
    )
    params_pspecs = _get_param_pspecs()

    def to_xla_sharding(pspecs):
      return jax.tree.map(lambda x: jax.sharding.NamedSharding(mesh, x), pspecs)

    partitioned_apply = jax.jit(
        transformed_apply,
        in_shardings=to_xla_sharding(
            (params_pspecs, jax.sharding.PartitionSpec('data'))
        ),
    )
    self.assertAllClose(
        jax.jit(transformed_apply)(params, images),
        partitioned_apply(params, images),
    )

    # Check gradients.
    @jax.grad
    def unpartitioned_grad(params, xs):
      return jnp.sum(jax.jit(transformed_apply)(params, xs))

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
      return jax2tf.convert(partitioned_apply)(params, inputs)

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
        transformed_apply(params, images), jax_fn(params, images)
    )
    self.assertAllClose(
        jax.jit(transformed_apply)(params, images),
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
    unpartitioned_output = jax.jit(transformed_apply)(params, images)
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
