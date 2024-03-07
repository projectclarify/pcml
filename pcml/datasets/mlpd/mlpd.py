from typing import Iterator, Tuple

import tensorflow_datasets as tfds
import tensorflow as tf

from collections import namedtuple

from absl import logging

import numpy as np

_DESCRIPTION = """Data from MLPD paper."""
_CITATION = """None"""

manifest = {
    'MLPD_high_negative': {
        'accn': 'SRR12960225',
        'affinity_category': '-3',
        'round': 'MLPD'
    },
    'MLPD_high_positive': {
        'accn': 'SRR12960226',
        'affinity_category': '3',
        'round': 'MLPD'
    },
    'MLPD_library': {
        'accn': 'SRR12960219',
        'affinity_category': 'NA',
        'round': 'MLPD'
    },
    'MLPD_low_negative': {
        'accn': 'SRR12960221',
        'affinity_category': '-1',
        'round': 'MLPD'
    },
    'MLPD_low_positive': {
        'accn': 'SRR12960222',
        'affinity_category': '1',
        'round': 'MLPD'
    },
    'MLPD_medium_negative': {
        'accn': 'SRR12960223',
        'affinity_category': '-2',
        'round': 'MLPD'
    },
    'MLPD_medium_positive': {
        'accn': 'SRR12960224',
        'affinity_category': '2',
        'round': 'MLPD'
    },
    'MLPD_very_high_negative': {
        'accn': 'SRR12960227',
        'affinity_category': '-4',
        'round': 'MLPD'
    },
    'MLPD_very_high_positive': {
        'accn': 'SRR12960228',
        'affinity_category': '4',
        'round': 'MLPD'
    },
    'round1_low_negative': {
        'accn': 'SRR12960220',
        'affinity_category': '-1',
        'round': '1'
    },
    'round1_low_positive': {
        'accn': 'SRR12960218',
        'affinity_category': '1',
        'round': '1'
    },
    'round1_medium_negative': {
        'accn': 'SRR12960229',
        'affinity_category': '-2',
        'round': '1'
    },
    'round1_medium_positive': {
        'accn': 'SRR12960230',
        'affinity_category': '2',
        'round': '1'
    },
    'round2_high_negative': {
        'accn': 'SRR12960216',
        'affinity_category': '-3',
        'round': '2'
    },
    'round2_high_positive': {
        'accn': 'SRR12960217',
        'affinity_category': '3',
        'round': '2'
    },
    'round2_low_negative': {
        'accn': 'SRR12960212',
        'affinity_category': '-1',
        'round': '2'
    },
    'round2_low_positive': {
        'accn': 'SRR12960213',
        'affinity_category': '1',
        'round': '2'
    },
    'round2_medium_negative': {
        'accn': 'SRR12960214',
        'affinity_category': '-2',
        'round': '2'
    },
    'round2_medium_positive': {
        'accn': 'SRR12960215',
        'affinity_category': '2',
        'round': '2'
    }
}


def _build_dataset_info(builder):
  return tfds.core.DatasetInfo(
      builder=builder,
      description=_DESCRIPTION,
      features=tfds.features.FeaturesDict({
          'sequence': tfds.features.Sequence(tf.int32),
          'affinity_category': tf.float32,
          'structure_x': tfds.features.Sequence(tf.float32),
          'structure_y': tfds.features.Sequence(tf.float32),
          'structure_energy': tf.float32,
      }),
      supervised_keys=None,
      homepage='https://github.com/cayley-group/flatland',
      citation=_CITATION,
  )


def _build_split_generators(train_paths, test_paths, validation_paths):
  return [
      tfds.core.SplitGenerator(
          name=tfds.Split.TRAIN,
          gen_kwargs={'filepaths': train_paths},
      ),
      tfds.core.SplitGenerator(
          name=tfds.Split.TEST,
          gen_kwargs={'filepaths': test_paths},
      ),
      tfds.core.SplitGenerator(
          name=tfds.Split.VALIDATION,
          gen_kwargs={'filepaths': validation_paths},
      ),
  ]


ExampleConfig = namedtuple("ExampleConfig", ["polymer_length", "alphabet_size"])


class Mlpd(tfds.core.GeneratorBasedBuilder):

  VERSION = tfds.core.Version('0.0.1')

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return _build_dataset_info(builder=self)

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Download the data and define splits."""
    return _build_split_generators(train_paths=None,
                                   test_paths=None,
                                   validation_paths=None)

  def example_config(self):
    return ExampleConfig(alphabet_size=4, polymer_length=10)

  def _generate_examples(self, filepaths) -> Iterator[Tuple[str, dict]]:
    """Generator of examples for each split."""

    logging.debug("Processing filepaths: %s" % filepaths)

    cfg = self.example_config()
    alphabet_size = cfg.alphabet_size
    polymer_length = cfg.polymer_length
    distances_shape = (polymer_length**2,)
    num_examples = 100

    for i in range(num_examples):
      yield str(i), {
          'sequence': np.random.randint(0, alphabet_size, (polymer_length,)),
          'affinity_category': np.random.random(),
          'structure_x': np.random.random((polymer_length,)),
          'structure_y': np.random.random((polymer_length,)),
          'structure_energy': np.random.random()
      }
