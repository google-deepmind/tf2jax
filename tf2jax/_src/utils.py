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
"""Util functions."""

import inspect


def fullargspec_to_signature(fullargspec) -> inspect.Signature:
  """Convert a {inspect|tf}.FullArgSpec to a inspect.Signature."""
  default_offset = len(fullargspec.args) - len(fullargspec.defaults or ())
  parameters = []
  # Positional or keyword args.
  for idx, arg in enumerate(fullargspec.args):
    param_dict = dict(name=arg, kind=inspect.Parameter.POSITIONAL_OR_KEYWORD)
    if idx >= default_offset:
      param_dict["default"] = fullargspec.defaults[idx - default_offset]
    parameters.append(inspect.Parameter(**param_dict))
  # Varargs
  if fullargspec.varargs is not None:
    parameters.append(
        inspect.Parameter(
            fullargspec.varargs, kind=inspect.Parameter.VAR_POSITIONAL))
  # Keyword-only args.
  for arg in fullargspec.kwonlyargs:
    param_dict = dict(name=arg, kind=inspect.Parameter.KEYWORD_ONLY)
    if arg in (fullargspec.kwonlydefaults or {}):
      param_dict["default"] = fullargspec.kwonlydefaults[arg]
    parameters.append(inspect.Parameter(**param_dict))
  # Kwargs.
  if fullargspec.varkw is not None:
    parameters.append(
        inspect.Parameter(
            fullargspec.varkw, kind=inspect.Parameter.VAR_KEYWORD))
  signature = inspect.Signature(parameters)

  return signature
