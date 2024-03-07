from typing import Iterator, Tuple

import tensorflow_datasets as tfds
import tensorflow as tf

from collections import namedtuple

from absl import logging

import numpy as np

import RNA

_DESCRIPTION = """Predict energies of random RNAs as estimated by ViennaRNA."""
_CITATION = """None"""


def _build_dataset_info(builder):
  return tfds.core.DatasetInfo(
      builder=builder,
      description=_DESCRIPTION,
      features=tfds.features.FeaturesDict({
          'sequence': tfds.features.Sequence(tf.int32),
          'structure_encoding': tfds.features.Sequence(tf.float32),
          'structure_energy': tf.float32,
      }),
      supervised_keys=None,
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


def _vector_encode_rna_structure(structure):

  encoding = {".": 0, "(": 1, ")": 2}

  encoded = np.zeros((len(structure),))

  for i, c in enumerate(structure):
    encoded[i] = encoding[c]

  return encoded


def _seq_vector_to_aucg(sequence_vector):

  encoding = ["A", "U", "C", "G"]

  output = ["N" for _ in range(len(sequence_vector))]

  for i, c in enumerate(sequence_vector):
    output[i] = encoding[c]

  return "".join(output)


class Vienna(tfds.core.GeneratorBasedBuilder):

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
    # Note: If the polymer length is too short, e.g. 10
    # the polymers will all be computed to have zero energy and no secondary
    # structure. A length of 30 yields non-uniform outputs.
    return ExampleConfig(alphabet_size=4, polymer_length=30)

  def _generate_examples(self, filepaths) -> Iterator[Tuple[str, dict]]:
    """Generator of examples for each split."""

    logging.debug("Processing filepaths: %s" % filepaths)

    cfg = self.example_config()
    alphabet_size = cfg.alphabet_size
    polymer_length = cfg.polymer_length
    distances_shape = (polymer_length**2,)
    num_examples = 100000

    for i in range(num_examples):

      sequence_vector = np.random.randint(0, alphabet_size, (polymer_length,))
      sequence_aucg = _seq_vector_to_aucg(sequence_vector)
      (structure_encoding, structure_free_energy) = RNA.fold(sequence_aucg)

      structure_vectorized = _vector_encode_rna_structure(structure_encoding)

      yield str(i), {
          'sequence': sequence_vector,
          'structure_encoding': structure_vectorized,
          'structure_energy': structure_free_energy
      }
