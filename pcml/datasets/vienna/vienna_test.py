import tensorflow as tf

import tensorflow_datasets as tfds

from polymers.datasets import vienna


def test_dataset_runs():

  ds = tfds.load('vienna', split="train")
  assert isinstance(ds, tf.data.Dataset)

  ds = ds.cache().repeat()
  for example in tfds.as_numpy(ds):
    print(example)
    break

  assert isinstance(example, dict)


if __name__ == "__main__":
  test_dataset_runs()
