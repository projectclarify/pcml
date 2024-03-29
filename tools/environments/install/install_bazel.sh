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

BAZEL_VERSION="2.0.0"

export SRC=$HOME/bin

mkdir -p $SRC

cd $SRC

SCRIPT="bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh"

wget https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/$SCRIPT

chmod +x $SCRIPT

./$SCRIPT --user

rm $SRC/$SCRIPT
