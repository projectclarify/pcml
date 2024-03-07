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

import uuid

import tensorflow as tf

from antidote.api.initialize import deploy
from antidote.utils.cfg_utils import Config

TEST_CONFIG = Config()


class TestDeploy(tf.test.TestCase):

  def test_deploy(self):

    staging = "%s/%s" % (TEST_CONFIG.get("test_artifacts_root"),
                         str(uuid.uuid4()))

    logs = deploy._deploy(project_id=TEST_CONFIG.get("project"),
                          region=TEST_CONFIG.get("region"))

    self.assertTrue("status: ACTIVE" in logs)


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.test.main()
