#!/usr/bin/env python
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
"""Test batch run wrapper."""

import datetime
from absl.testing import absltest
from absl import logging

import os

from antidote.batch import fyre
from antidote.batch import jobs

from antidote.test_config import TEST_CONFIG


class MainTest(absltest.TestCase):

  def setUp(self):

    if "ANTIDOTE_WORKSPACE_ROOT" not in os.environ:
      logging.error("Requires ANTIDOTE_WORKSPACE_ROOT to be set.")

    self.workspace_root = os.environ["ANTIDOTE_WORKSPACE_ROOT"]
    self.project = TEST_CONFIG["project"]
    self.runtime_root = TEST_CONFIG["runtime_antidote_root"]

  def test_basic(self):

    config_path = os.path.join(
        self.runtime_root, "antidote/research/mnist/configs/test_config.py")

    cmd = [
        "python", "-m", "antidote.research.mnist.main", "--workdir=/tmp/mnist",
        "--config=%s" % config_path
    ]

    f = fyre.Fyre(workspace_root=self.workspace_root,
                  project=self.project,
                  command=cmd)

    create_response = f.batch_run()

    status_response = f.wait_for_job(
        timeout=datetime.timedelta(seconds=3600),
        polling_interval=datetime.timedelta(seconds=1))

    assert status_response.status.succeeded == 1


if __name__ == '__main__':
  absltest.main()
