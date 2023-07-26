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
"""MHLO primitive tests."""

import functools

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized

import chex
import jax
import numpy as np

from tf2jax.experimental import mhlo


def _convert_to_mhlo(jax_fn, inputs, *, dialect):
  lowered_forward = jax_fn.lower(*inputs)
  mhlo_text = lowered_forward.as_text(dialect=dialect)
  return mhlo_text


def _check_transforms(fn, inputs, *, dialect):
  jaxpr = jax.make_jaxpr(fn)(*inputs)
  logging.info(jaxpr)

  mhlo_text = jax.jit(fn).lower(*inputs).as_text(dialect=dialect)
  logging.info(mhlo_text)


class MhloTest(chex.TestCase):

  def _assert_all_close(self, expect_fn, actual_fn, inputs):
    expect_outputs = self.variant(expect_fn)(*inputs)
    actual_outputs = self.variant(actual_fn)(*inputs)
    logging.info(expect_outputs)
    logging.info(actual_outputs)
    chex.assert_trees_all_close(expect_outputs, actual_outputs)

  @chex.variants(with_jit=True, without_jit=True)
  @parameterized.named_parameters(
      ("mhlo", "mhlo"),
      ("stablehlo", "stablehlo"),
  )
  def test_one_input_and_one_output(self, dialect):
    @jax.jit
    def fn(x):
      return x * 2 + 3.14

    inputs = (np.ones((3, 2), dtype=np.float32) * 10,)
    mhlo_text = _convert_to_mhlo(
        fn, jax.tree_map(np.zeros_like, inputs), dialect=dialect)
    mhlo_module = mhlo.MhloModule(module=mhlo_text, fun_name="test_module")

    chex.assert_trees_all_close(
        mhlo.mhlo_apply(*inputs, module=mhlo_module), fn(*inputs))

    def make_top_fn(sub_fn):
      def top_fn(x):
        return sub_fn(x + 8) * 10
      return top_fn

    expect_top_fn = make_top_fn(fn)
    actual_top_fn = make_top_fn(
        functools.partial(mhlo.mhlo_apply, module=mhlo_module))
    self._assert_all_close(expect_top_fn, actual_top_fn, inputs)
    _check_transforms(actual_top_fn, inputs, dialect=dialect)

  @chex.variants(with_jit=True, without_jit=True)
  @parameterized.named_parameters(
      ("mhlo", "mhlo"),
      ("stablehlo", "stablehlo"),
  )
  def test_two_inputs_and_one_output(self, dialect):
    @jax.jit
    def fn(x, y):
      return x * 2 + y

    inputs = (
        np.ones((3, 2), dtype=np.float32) * 10,
        np.ones((3, 2), dtype=np.float32) * 20,
    )
    mhlo_text = _convert_to_mhlo(
        fn, jax.tree_map(np.zeros_like, inputs), dialect=dialect)
    mhlo_module = mhlo.MhloModule(module=mhlo_text, fun_name="test_module")
    chex.assert_trees_all_close(
        mhlo.mhlo_apply(*inputs, module=mhlo_module), fn(*inputs))

    def make_top_fn(sub_fn):
      def top_fn(x, y):
        return sub_fn(x + 8, y + 9) * 10
      return top_fn

    expect_top_fn = make_top_fn(fn)
    actual_top_fn = make_top_fn(
        functools.partial(mhlo.mhlo_apply, module=mhlo_module))
    self._assert_all_close(expect_top_fn, actual_top_fn, inputs)
    _check_transforms(actual_top_fn, inputs, dialect=dialect)

  @chex.variants(with_jit=True, without_jit=True)
  @parameterized.named_parameters(
      ("mhlo", "mhlo"),
      ("stablehlo", "stablehlo"),
  )
  def test_two_inputs_and_two_outputs(self, dialect):
    @jax.jit
    def fn(x, y):
      return x * 2 + y, x + y * 3

    inputs = (
        np.ones((3, 2), dtype=np.float32) * 10,
        np.ones((3, 2), dtype=np.float32) * 20,
    )
    mhlo_text = _convert_to_mhlo(
        fn, jax.tree_map(np.zeros_like, inputs), dialect=dialect)
    mhlo_module = mhlo.MhloModule(module=mhlo_text, fun_name="test_module")
    chex.assert_trees_all_close(
        mhlo.mhlo_apply(*inputs, module=mhlo_module), fn(*inputs))

    def make_top_fn(sub_fn):
      def top_fn(x, y):
        res0, res1 = sub_fn(x + 8, y + 9)
        return res0 * 10, res1 * 3.14
      return top_fn

    expect_top_fn = make_top_fn(fn)
    actual_top_fn = make_top_fn(
        functools.partial(mhlo.mhlo_apply, module=mhlo_module))
    self._assert_all_close(expect_top_fn, actual_top_fn, inputs)
    _check_transforms(actual_top_fn, inputs, dialect=dialect)

  @chex.variants(with_jit=True, without_jit=True)
  @parameterized.named_parameters(
      ("mhlo", "mhlo"),
      ("stablehlo", "stablehlo"),
  )
  def test_boolean(self, dialect):
    @jax.jit
    def fn(x):
      return x > 0

    inputs = (
        np.ones((4,), dtype=np.int32) * 10,
    )
    mhlo_text = _convert_to_mhlo(
        fn, jax.tree_map(np.zeros_like, inputs), dialect=dialect)
    mhlo_module = mhlo.MhloModule(module=mhlo_text, fun_name="test_module")
    chex.assert_trees_all_close(
        mhlo.mhlo_apply(*inputs, module=mhlo_module), fn(*inputs))

    def make_top_fn(sub_fn):
      def top_fn(x):
        return sub_fn(x) * x
      return top_fn

    expect_top_fn = make_top_fn(fn)
    actual_top_fn = make_top_fn(
        functools.partial(mhlo.mhlo_apply, module=mhlo_module))
    self._assert_all_close(expect_top_fn, actual_top_fn, inputs)


if __name__ == "__main__":
  absltest.main()
