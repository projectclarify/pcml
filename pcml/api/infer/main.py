from flask import escape, abort
import datetime

from antidote.research.mnist import train

from flax.training import checkpoints
from antidote.research.mnist.configs import test_config
import jax

from jax import numpy as np

from jax import random

from absl import logging
logging.set_verbosity(logging.DEBUG)

workdir = "./"

config = test_config.get_config()

state = train.create_train_state(
  jax.random.PRNGKey(0), config)

state = checkpoints.restore_checkpoint(workdir, state)


def infer(request):
  """HTTP Cloud Function.
    Args:
        request (flask.Request): The request object.
        <https://flask.palletsprojects.com/en/1.1.x/api/#incoming-request-data>
    Returns:
        The response text, or any set of values that can be turned into a
        Response object using `make_response`
        <https://flask.palletsprojects.com/en/1.1.x/api/#flask.make_response>.
  """

  # Set CORS headers for preflight requests
  if request.method == 'OPTIONS':
    # Allows GET requests from origin https://mydomain.com with
    # Authorization header
    headers = {
        # Change the origin to the application origin domain
        'Access-Control-Allow-Origin': '*googleusercontent.com',
        'Access-Control-Allow-Methods': 'POST',
        'Access-Control-Allow-Headers': 'Authorization',
        'Access-Control-Max-Age': '3600',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Allow-Credentials': 'true'
    }
    return ('', 204, headers)

  # Set CORS headers for main requests
  headers = {
    'Access-Control-Allow-Origin': '*.googleusercontent.com',
    'Access-Control-Allow-Credentials': 'true'
  }

  if request.method != 'POST':
    return abort(403)

  content_type = request.headers['content-type']
  if content_type != 'application/json':
    return abort(403)

  request_json = request.get_json(silent=True)
  if request_json and 'images' in request_json:
    image_data = np.asarray(request_json['images'])

  logits = train.CNN().apply({'params': state.params}, image_data)
  predictions = list(logits.argmax(axis=1))

  #json.dumps({"predictions": predictions})

  return "Hello {}, {}, {}".format(predictions[0], predictions[1], predictions[2])