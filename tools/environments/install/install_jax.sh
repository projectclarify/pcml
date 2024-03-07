# coding=utf-8
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

export LIB_ROOT=/usr/local/src

mkdir -p ${LIB_ROOT} && cd ${LIB_ROOT}

git clone https://github.com/google/jax.git

cd ${LIB_ROOT}/jax && git checkout 29db4203fec6b9796637034d48a62ac84fbcf43f

python ${LIB_ROOT}/jax/build/build.py

ln -s ${LIB_ROOT}/jax/build ${LIB_ROOT}/jaxlib
