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
"""Utils.

A subset of functions provided in
https://github.com/google/jax/blob/main/jax/_src/util.py
"""

import itertools as it
from typing import Iterable, List, Sequence, TypeVar


T = TypeVar("T")


def concatenate(xs: Iterable[Sequence[T]]) -> List[T]:
  """Concatenates/flattens a list of lists."""
  return list(it.chain.from_iterable(xs))

flatten = concatenate

_UNFLATTEN_DONE = object()


def unflatten(xs: Iterable[T], ns: Sequence[int]) -> List[List[T]]:
  """Splits `xs` into subsequences of lengths `ns`."""
  xs_iter = iter(xs)
  unflattened = [[next(xs_iter) for _ in range(n)] for n in ns]
  assert next(xs_iter, _UNFLATTEN_DONE) is _UNFLATTEN_DONE
  return unflattened
