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
"""TF2JAX configurations."""

import contextlib

from typing import Any, Union

_config = dict(
    strict_shape_check=True,
    strict_dtype_check=False,
    force_const_float32_to_bfloat16=False,
    force_const_float64_to_bfloat16=False,
    convert_custom_gradient=True,
    infer_relu_from_jax2tf=True,
    infer_cumulative_reduction_from_jax2tf=True,
    raise_on_prevent_gradient=True,
    # TensorFlow uses high/highest precision on TPU.
    resize_bilinear_precision="highest",
    # This is temporary until checkify supports debug mode.
    enable_checkify_for_asserts=False,
    # Enable checks that each natively serialized computation is executed on a
    # platform for which it was lowered.
    xlacallmodule_strict_checks=True,
)


def get_config(name: str) -> Union[bool, str]:
  return _config[name]


def update_config(name: str, value: Any):
  if name in _config:
    _config[name] = value
  else:
    raise ValueError(
        f"Parameter named {name} not found in config={_config}")


@contextlib.contextmanager
def override_config(name: str, value: Any):
  old_value = get_config(name)
  update_config(name, value)
  try:
    yield
  finally:
    update_config(name, old_value)
