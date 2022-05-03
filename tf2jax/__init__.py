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
"""API of tf2jax."""

from tf2jax._src.tf2jax import convert
from tf2jax._src.tf2jax import convert_from_restored
from tf2jax._src.tf2jax import convert_functional
from tf2jax._src.tf2jax import convert_functional_from_restored

from tf2jax._src.tf2jax import get_config
from tf2jax._src.tf2jax import override_config
from tf2jax._src.tf2jax import update_config

__version__ = "0.2.0"

#  _________________________________________
# / Please don't use symbols in `_src` they \
# \ are not part of the tf2jax public API.    /
#  -----------------------------------------
#         \   ^__^
#          \  (oo)\_______
#             (__)\       )\/\
#                 ||----w |
#                 ||     ||
