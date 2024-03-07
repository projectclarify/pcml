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


def _deploy(project_id,
            checkpoint_path,
            service_account=None,
            source=None,
            runtime="python37",
            region="us-central1",
            memory="8192MB",
            timeout="540s",
            max_instances=1000):

  function_name = "infer"

  lib_root = get_project_root()

  this_dir = os.path.join(lib_root, "antidote", "api", "infer")

  with TemporaryDirectory() as tmpdir:

    # Copy in library code
    subprocess.check_output(["cp", "-r", lib_root, tmpdir])

    tmp_libdir = os.path.join(tmpdir, LIB_NAME)

    # Copy in function code
    source = os.path.join(this_dir, "main.py")
    target = os.path.join(tmp_libdir, "main.py")
    tf.io.gfile.copy(source, target)

    filename = checkpoint_path.split("/")[-1]
    target = os.path.join(tmp_libdir, filename)
    tf.io.gfile.copy(checkpoint_path, target)

    # hack
    events_filename = "events.out.tfevents.1628202753.python-20210609-171639.2916498.137.v2"
    source_dir = "/tmp/mnist2"
    events_path = os.path.join(source_dir, events_filename)
    target = os.path.join(tmp_libdir, events_filename)
    tf.io.gfile.copy(events_path, target)

    cmd = [
        "gcloud", "functions", "deploy", function_name, "--trigger-http",
        "--runtime", runtime, "--region", region,
        "--memory", memory, "--timeout", timeout
    ]

    return run_and_output(cmd, cwd=tmp_libdir)


def main(_):

  _deploy(project_id=FLAGS.project,
          service_account=FLAGS.service_account,
          region=FLAGS.gcp_region,
          checkpoint_path=FLAGS.checkpoint_path)


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

  flags.DEFINE_string("checkpoint_path", None,
                      "Path to a checkpoint file to restore.")

  logging.set_verbosity(logging.DEBUG)
  app.run(main)
