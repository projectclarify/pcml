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
"""Test config.

Modify this for local development if you don't have access to the
listed resources and omit this file when sending a pull request.

"""

TEST_CONFIG = {
    "service_account": "/home/jovyan/key.json",
    "project": "clarify",
    "test_artifacts_root": "gs://clarify-dev2/test",
    "region": "us-central1",
    "zone": "us-central1-a",
    "functions_testing_service_account": "clarify@appspot.gserviceaccount.com",
    "runtime_antidote_root": "/usr/local/src/antidote"
}
