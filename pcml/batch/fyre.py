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
"""Batch run wrapper."""

from absl import logging
import subprocess
import sys
import os
import uuid

import kubernetes.client
from kubernetes import client

from antidote.batch import job

from antidote.utils.cmd_utils import run_and_output


class Fyre(object):

  def __init__(self,
               command,
               workspace_root,
               project,
               job=job.SimpleJob,
               extra_container_args={},
               extra_job_args={}):

    os.chdir(workspace_root)

    logging.info("Building workspace at {}...".format(workspace_root))

    dockerfile_path = os.path.join(workspace_root, "tools", "environments",
                                   "Dockerfile.rtbase")

    image_tag = "gcr.io/%s/runtime:%s" % (project, str(uuid.uuid4()))

    # ---
    # Build the job runtime
    logging.info("Baking runtime...")
    run_and_output(
        ["docker", "build", "-t", image_tag, "-f", dockerfile_path, "."],
        cwd=workspace_root)

    logging.info("Pushing runtime...")
    push_output = run_and_output(["gcloud", "docker", "--", "push", image_tag],
                                 cwd=workspace_root)

    container = client.V1Container(name="fyre",
                                   image=image_tag,
                                   command=command,
                                   **extra_container_args)

    self.job_object = job(container=container, **extra_job_args)

    logging.info("Finished initializing Fyre job.")

  def batch_run(self):
    return self.job_object.batch_run()

  def wait_for_job(self, *args, **kwargs):
    return self.job_object.wait_for_job(*args, **kwargs)
