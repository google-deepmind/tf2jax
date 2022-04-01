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

# Runs CI tests on a local machine.
set -xeuo pipefail

# Install deps in a virtual env.
readonly VENV_DIR=/tmp/tf2jax-env
rm -rf "${VENV_DIR}"
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
python --version

# Install dependencies.
pip install --upgrade pip setuptools wheel
pip install flake8 pytest-xdist pytype pylint pylint-exit
pip install -r requirements.txt
pip install -r requirements_tests.txt

# Lint with flake8.
flake8 `find tf2jax -name '*.py' | xargs` --count --select=E9,F63,F7,F82,E225,E251 --show-source --statistics

# Lint with pylint.
PYLINT_ARGS="-efail -wfail"
# Lint modules and tests separately.
python -m pylint --rcfile=.pylintrc `find tf2jax -name '*.py' | grep -v 'test.py' | xargs` || pylint-exit $PYLINT_ARGS $?
# Disable `protected-access` warnings for tests.
# Disable `unexpected-keyword-arg` and `no-value-for-parameter` error for tests due to false positives in tensorflow.
python -m pylint --rcfile=.pylintrc `find tf2jax -name '*_test.py' | xargs` -d W0212,E1123,E1120 || pylint-exit $PYLINT_ARGS $?

# Build the package.
python setup.py sdist
pip wheel --verbose --no-deps --no-clean dist/tf2jax*.tar.gz
pip install tf2jax*.whl

# Check types with pytype.
# Also see https://github.com/google/pytype/issues/1169
pytype `find tf2jax/_src/ -name "*py" | xargs` -k --use-enum-overlay

# Run tests using pytest.
# Change directory to avoid importing the package from repo root.
mkdir _testing && cd _testing

# Main tests.
# CPU count on macos or linux
if [ "$(uname)" == "Darwin" ]; then
  N_JOBS=$(sysctl -n hw.logicalcpu)
else
  N_JOBS=$(grep -c ^processor /proc/cpuinfo)
fi

pytest -n "${N_JOBS}" --pyargs tf2jax
cd ..

set +u
deactivate
echo "All tests passed. Congrats!"
