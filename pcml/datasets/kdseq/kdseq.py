from typing import Iterator, Tuple

import tensorflow_datasets as tfds
import tensorflow as tf

from collections import namedtuple

from absl import logging

import numpy as np

_DESCRIPTION = """Unpublished Kd-seq data."""
_CITATION = """None"""

_MANIFEST = {
    '20201205_DlldN-10B': {
        'accn': 'SRR14703142',
        'bound_unbound': 'bound',
        'dna_conc': '20nM',
        'round': '1',
        'tf_conc': '1000nM'
    },
    '20201205_DlldN-10UB': {
        'accn': 'SRR14703143',
        'bound_unbound': 'unbound',
        'dna_conc': '20nM',
        'round': '1',
        'tf_conc': '1000nM'
    },
    '20201205_DlldN-11B': {
        'accn': 'SRR14703144',
        'bound_unbound': 'bound',
        'dna_conc': '20nM',
        'round': '1',
        'tf_conc': '330nM'
    },
    '20201205_DlldN-11UB': {
        'accn': 'SRR14703145',
        'bound_unbound': 'unbound',
        'dna_conc': '20nM',
        'round': '1',
        'tf_conc': '330nM'
    },
    '20201205_DlldN-12B': {
        'accn': 'SRR14703146',
        'bound_unbound': 'bound',
        'dna_conc': '20nM',
        'round': '1',
        'tf_conc': '100nM'
    },
    '20201205_DlldN-12UB': {
        'accn': 'SRR14703147',
        'bound_unbound': 'unbound',
        'dna_conc': '20nM',
        'round': '1',
        'tf_conc': '100nM'
    },
    '20201205_DlldN-1B': {
        'accn': 'SRR14703124',
        'bound_unbound': 'bound',
        'dna_conc': '500nM',
        'round': '1',
        'tf_conc': '3300nM'
    },
    '20201205_DlldN-1UB': {
        'accn': 'SRR14703125',
        'bound_unbound': 'unbound',
        'dna_conc': '500nM',
        'round': '1',
        'tf_conc': '3300nM'
    },
    '20201205_DlldN-2B': {
        'accn': 'SRR14703126',
        'bound_unbound': 'bound',
        'dna_conc': '500nM',
        'round': '1',
        'tf_conc': '1000nM'
    },
    '20201205_DlldN-2UB': {
        'accn': 'SRR14703127',
        'bound_unbound': 'unbound',
        'dna_conc': '500nM',
        'round': '1',
        'tf_conc': '1000nM'
    },
    '20201205_DlldN-3B': {
        'accn': 'SRR14703128',
        'bound_unbound': 'bound',
        'dna_conc': '500nM',
        'round': '1',
        'tf_conc': '330nM'
    },
    '20201205_DlldN-3UB': {
        'accn': 'SRR14703129',
        'bound_unbound': 'unbound',
        'dna_conc': '500nM',
        'round': '1',
        'tf_conc': '330nM'
    },
    '20201205_DlldN-4B': {
        'accn': 'SRR14703130',
        'bound_unbound': 'bound',
        'dna_conc': '500nM',
        'round': '1',
        'tf_conc': '100nM'
    },
    '20201205_DlldN-4UB': {
        'accn': 'SRR14703131',
        'bound_unbound': 'unbound',
        'dna_conc': '500nM',
        'round': '1',
        'tf_conc': '100nM'
    },
    '20201205_DlldN-5B': {
        'accn': 'SRR14703132',
        'bound_unbound': 'bound',
        'dna_conc': '100nM',
        'round': '1',
        'tf_conc': '3300nM'
    },
    '20201205_DlldN-5UB': {
        'accn': 'SRR14703133',
        'bound_unbound': 'unbound',
        'dna_conc': '100nM',
        'round': '1',
        'tf_conc': '3300nM'
    },
    '20201205_DlldN-6B': {
        'accn': 'SRR14703134',
        'bound_unbound': 'bound',
        'dna_conc': '100nM',
        'round': '1',
        'tf_conc': '1000nM'
    },
    '20201205_DlldN-6UB': {
        'accn': 'SRR14703135',
        'bound_unbound': 'unbound',
        'dna_conc': '100nM',
        'round': '1',
        'tf_conc': '1000nM'
    },
    '20201205_DlldN-7B': {
        'accn': 'SRR14703136',
        'bound_unbound': 'bound',
        'dna_conc': '100nM',
        'round': '1',
        'tf_conc': '330nM'
    },
    '20201205_DlldN-7UB': {
        'accn': 'SRR14703137',
        'bound_unbound': 'unbound',
        'dna_conc': '100nM',
        'round': '1',
        'tf_conc': '330nM'
    },
    '20201205_DlldN-8B': {
        'accn': 'SRR14703138',
        'bound_unbound': 'bound',
        'dna_conc': '100nM',
        'round': '1',
        'tf_conc': '100nM'
    },
    '20201205_DlldN-8UB': {
        'accn': 'SRR14703139',
        'bound_unbound': 'unbound',
        'dna_conc': '100nM',
        'round': '1',
        'tf_conc': '100nM'
    },
    '20201205_DlldN-9B': {
        'accn': 'SRR14703140',
        'bound_unbound': 'bound',
        'dna_conc': '20nM',
        'round': '1',
        'tf_conc': '3300nM'
    },
    '20201205_DlldN-9UB': {
        'accn': 'SRR14703141',
        'bound_unbound': 'unbound',
        'dna_conc': '20nM',
        'round': '1',
        'tf_conc': '3300nM'
    },
    '20201205_R0': {
        'accn': 'SRR14703123',
        'bound_unbound': 'None',
        'dna_conc': 'NA',
        'round': '0',
        'tf_conc': 'NA'
    }
}


def _build_dataset_info(builder):
  return tfds.core.DatasetInfo(
      builder=builder,
      description=_DESCRIPTION,
      features=tfds.features.FeaturesDict({
          'sequence': tfds.features.Sequence(tf.int32),
          'affinity': tf.float32,
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


class KdSeq(tfds.core.GeneratorBasedBuilder):

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
          'sequence': np.random.randint(0, alphabet_size, (olymer_length,)),
          'affinity': np.random.random(),
          'structure_x': np.random.random((polymer_length,)),
          'structure_y': np.random.random((polymer_length,)),
          'structure_energy': np.random.random()
      }
