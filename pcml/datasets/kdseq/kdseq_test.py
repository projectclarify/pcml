import tensorflow as tf

import tensorflow_datasets as tfds

from polymers.datasets import kdseq


def test_dataset_runs():

  ds = tfds.load('kd_seq', split="train")
  assert isinstance(ds, tf.data.Dataset)

  ds = ds.cache().repeat()
  for example in tfds.as_numpy(ds):
    break

  assert isinstance(example, dict)
