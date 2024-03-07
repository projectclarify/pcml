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

import os
import datetime
import tempfile
import subprocess

import tensorflow as tf

from antidote.api.utils import deployment_utils
from antidote.utils.fs_utils import get_project_root
from antidote.utils.fs_utils import TemporaryDirectory
from antidote.utils.cmd_utils import run_and_output

LIB_NAME = "antidote"


def _touch(path):
  with open(path, "w") as f:
    f.write("")


def _timestamp():
  now = datetime.datetime.now()
  epoch = datetime.datetime.utcfromtimestamp(0)
  ts = int((now - epoch).total_seconds() * 100000.0)
  return ts


def _deploy(project_id,
            service_account=None,
            source=None,
            runtime="python37",
            region="us-central1",
            memory="256MB",
            timeout="60s",
            max_instances=1000):

  function_name = "initialize"

  cmd = [
      "gcloud", "functions", "deploy", function_name, "--trigger-http",
      "--runtime", runtime, "--region", region, "--allow-unauthenticated"
  ]

  return run_and_output(cmd)


def main(_):

  _deploy(project_id=FLAGS.project,
          service_account=FLAGS.service_account,
          region=FLAGS.gcp_region)


if __name__ == "__main__":

  from absl import app
  from absl import flags
  from absl import logging

  FLAGS = flags.FLAGS

  flags.DEFINE_string("project", "clarify", "A project ID.")

  flags.DEFINE_string("service_account", "clarify@appspot.gserviceaccount.com",
                      "A service acct to allow GCF to r/w GCP resources.")

  flags.DEFINE_string("staging_root", "gs://clarify-dev/tmp/fnstaging",
                      "GCS bucket to use for staging function bundles.")

  flags.DEFINE_string("gcp_region", "us-central1",
                      "A GCP region where the function will be deployed.")

  logging.set_verbosity(logging.DEBUG)
  app.run(main)
