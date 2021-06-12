# Copyright 2019 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""A demo data set constructed with MediaSequence and MediaPipe.

This code demonstrates the steps for constructing a data set with MediaSequence.
This code has two functions. First, it can be run as a module to download and
prepare a toy dataset. Second, it can be imported and used to provide a
tf.data.Dataset reading that data from disk via as_dataset().

Running as a module prepares the data in three stages via generate_examples().
First, the actual data files are downloaded. If the download is disrupted, the
incomplete files will need to be removed before running the script again.
Second, the annotations are parsed and reformated into metadata as described in
the MediaSequence documentation. Third, MediaPipe is run to extract subsequences
of frames for subsequent training via _run_mediapipe().

The toy data set is classifying a clip as a panning shot of galaxy or nebula
from videos releasued under the [Creative Commons Attribution 4.0 International
license](http://creativecommons.org/licenses/by/4.0/) on the ESA/Hubble site.
(The use of these ESA/Hubble materials does not imply the endorsement by
ESA/Hubble or any ESA/Hubble employee of a commercial product or service.) Each
video is split into 5 or 6 ten-second clips with a label of "galaxy" or "nebula"
and downsampled to 10 frames per second. (The last clip for each test example is
only 6 seconds.) There is one video of each class in each of the training and
testing splits.

Reading the data as a tf.data.Dataset is accomplished with the following lines:

   demo = DemoDataset("demo_data_path")
   dataset = demo.as_dataset("test")
   # implement additional processing and batching here
   images_and_labels = dataset.make_one_shot_iterator().get_next()
   images = images_and_labels["images"]
   labels = image_and_labels["labels"]
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import csv
import os
import random
import subprocess
import sys
import tempfile

from absl import app
from absl import flags
from absl import logging
from six.moves import range
from six.moves import urllib
import tensorflow.compat.v1 as tf

from mediapipe.util.sequence import media_sequence as ms

SPLITS = {
    "train":
        """url,label index,label string,duration,credits
https://cdn.spacetelescope.org/archives/videos/medium_podcast/heic1608c.mp4,0,nebula,50,"ESA/Hubble; Music: Johan B. Monell"
https://cdn.spacetelescope.org/archives/videos/medium_podcast/heic1712b.mp4,1,galaxy,50,"ESA/Hubble, Digitized Sky Survey, Nick Risinger (skysurvey.org) Music: Johan B Monell"
""",
    "test":
        """url,label index,label string,duration,credits
https://cdn.spacetelescope.org/archives/videos/medium_podcast/heic1301b.m4v,0,nebula,56,"NASA, ESA. Acknowledgement: Josh Lake"
https://cdn.spacetelescope.org/archives/videos/medium_podcast/heic1305b.m4v,1,galaxy,56,"NASA, ESA, Digitized Sky Survey 2. Acknowledgement: A. van der Hoeven"
"""
}
NUM_CLASSES = 2
NUM_SHARDS = 2
SECONDS_PER_EXAMPLE = 10
MICROSECONDS_PER_SECOND = 1000000
TF_RECORD_PATTERN = "demo_space_dataset_%s_tfrecord"
GRAPHS = ["clipped_images_from_file_at_24fps.pbtxt"]


class DemoDataset(object):
  """Generates and loads a demo data set."""

  def __init__(self, path_to_data):
    if not path_to_data:
      raise ValueError("You must supply the path to the data directory.")
    self.path_to_data = path_to_data

  def as_dataset(self,
                 split,
                 shuffle=False,
                 repeat=False,
                 serialized_prefetch_size=32,
                 decoded_prefetch_size=32):
    """Returns the dataset as a tf.data.Dataset.

    Args:
      split: either "train" or "test"
      shuffle: if true, shuffles both files and examples.
      repeat: if true, repeats the data set forever.
      serialized_prefetch_size: the buffer size for reading from disk.
      decoded_prefetch_size: the buffer size after decoding.

    Returns:
      A tf.data.Dataset object with the following structure: {
        "images": uint8 tensor, shape [time, height, width, channels]
        "labels": one hot encoded label tensor, shape [2]
        "id": a unique string id for each example, shape []
      }
    """

    def parse_fn(sequence_example):
      """Parses a clip classification example."""
      context_features = {
          ms.get_example_id_key():
              ms.get_example_id_default_parser(),
          ms.get_clip_label_index_key():
              ms.get_clip_label_index_default_parser(),
          ms.get_clip_label_string_key():
              ms.get_clip_label_string_default_parser()
      }
      sequence_features = {
          ms.get_image_encoded_key(): ms.get_image_encoded_default_parser(),
      }
      parsed_context, parsed_sequence = tf.io.parse_single_sequence_example(
          sequence_example, context_features, sequence_features)
      example_id = parsed_context[ms.get_example_id_key()]
      classification_target = tf.one_hot(
          tf.sparse_tensor_to_dense(
              parsed_context[ms.get_clip_label_index_key()]), NUM_CLASSES)
      images = tf.map_fn(
          tf.image.decode_jpeg,
          parsed_sequence[ms.get_image_encoded_key()],
          back_prop=False,
          dtype=tf.uint8)
      return {
          "id": example_id,
          "labels": classification_target,
          "images": images,
      }

    if split not in SPLITS:
      raise ValueError("split '%s' is unknown." % split)
    all_shards = tf.io.gfile.glob(
        os.path.join(self.path_to_data, TF_RECORD_PATTERN % split + "-*-of-*"))
    if shuffle:
      random.shuffle(all_shards)
    all_shards_dataset = tf.data.Dataset.from_tensor_slices(all_shards)
    cycle_length = min(16, len(all_shards))
    dataset = all_shards_dataset.apply(
        tf.data.experimental.parallel_interleave(
            tf.data.TFRecordDataset,
            cycle_length=cycle_length,
            block_length=1,
            sloppy=True,
            buffer_output_elements=serialized_prefetch_size))
    dataset = dataset.prefetch(serialized_prefetch_size)
    if shuffle:
      dataset = dataset.shuffle(serialized_prefetch_size)
    if repeat:
      dataset = dataset.repeat()
    dataset = dataset.map(parse_fn)
    dataset = dataset.prefetch(decoded_prefetch_size)
    return dataset

  def generate_examples(self, path_to_mediapipe_binary,
                        path_to_graph_directory):
    """Downloads data and generates sharded TFRecords.

    Downloads the data files, generates metadata, and processes the metadata
    with MediaPipe to produce tf.SequenceExamples for training. The resulting
    files can be read with as_dataset(). After running this function the
    original data files can be deleted.

    Args:
      path_to_mediapipe_binary: Path to the compiled binary for the BUILD target
        mediapipe/examples/desktop/demo:media_sequence_demo.
      path_to_graph_directory: Path to the directory with MediaPipe graphs in
        mediapipe/graphs/media_sequence/.
    """
    if not path_to_mediapipe_binary:
      raise ValueError("You must supply the path to the MediaPipe binary for "
                       "mediapipe/examples/desktop/demo:media_sequence_demo.")
    if not path_to_graph_directory:
      raise ValueError(
          "You must supply the path to the directory with MediaPipe graphs in "
          "mediapipe/graphs/media_sequence/.")
    logging.info("Downloading data.")
    tf.io.gfile.makedirs(self.path_to_data)
    if sys.version_info >= (3, 0):
      urlretrieve = urllib.request.urlretrieve
    else:
      urlretrieve = urllib.request.urlretrieve
    for split in SPLITS:
      reader = csv.DictReader(SPLITS[split].split("\n"))
      all_metadata = []
      for row in reader:
        url = row["url"]
        basename = url.split("/")[-1]
        local_path = os.path.join(self.path_to_data, basename)
        if not tf.io.gfile.exists(local_path):
          urlretrieve(url, local_path)

        for start_time in range(0, int(row["duration"]), SECONDS_PER_EXAMPLE):
          metadata = tf.train.SequenceExample()
          ms.set_example_id(bytes23(basename + "_" + str(start_time)),
                            metadata)
          ms.set_clip_data_path(bytes23(local_path), metadata)
          ms.set_clip_start_timestamp(start_time * MICROSECONDS_PER_SECOND,
                                      metadata)
          ms.set_clip_end_timestamp(
              (start_time + SECONDS_PER_EXAMPLE) * MICROSECONDS_PER_SECOND,
              metadata)
          ms.set_clip_label_index((int(row["label index"]),), metadata)
          ms.set_clip_label_string((bytes23(row["label string"]),),
                                   metadata)
          all_metadata.append(metadata)
      random.seed(47)
      random.shuffle(all_metadata)
      shard_names = [self._indexed_shard(split, i) for i in range(NUM_SHARDS)]
      writers = [tf.io.TFRecordWriter(shard_name) for shard_name in shard_names]
      with _close_on_exit(writers) as writers:
        for i, seq_ex in enumerate(all_metadata):
          for graph in GRAPHS:
            graph_path = os.path.join(path_to_graph_directory, graph)
            seq_ex = self._run_mediapipe(path_to_mediapipe_binary, seq_ex,
                                         graph_path)
          writers[i % len(writers)].write(seq_ex.SerializeToString())

  def _indexed_shard(self, split, index):
    """Constructs a sharded filename."""
    return os.path.join(
        self.path_to_data,
        TF_RECORD_PATTERN % split + "-%05d-of-%05d" % (index, NUM_SHARDS))

  def _run_mediapipe(self, path_to_mediapipe_binary, sequence_example, graph):
    """Runs MediaPipe over MediaSequence tf.train.SequenceExamples.

    Args:
      path_to_mediapipe_binary: Path to the compiled binary for the BUILD target
        mediapipe/examples/desktop/demo:media_sequence_demo.
      sequence_example: The SequenceExample with metadata or partial data file.
      graph: The path to the graph that extracts data to add to the
        SequenceExample.

    Returns:
      A copy of the input SequenceExample with additional data fields added
      by the MediaPipe graph.
    Raises:
      RuntimeError: if MediaPipe returns an error or fails to run the graph.
    """
    if not path_to_mediapipe_binary:
      raise ValueError("--path_to_mediapipe_binary must be specified.")
    input_fd, input_filename = tempfile.mkstemp()
    output_fd, output_filename = tempfile.mkstemp()
    cmd = [
        path_to_mediapipe_binary,
        "--calculator_graph_config_file=%s" % graph,
        "--input_side_packets=input_sequence_example=%s" % input_filename,
        "--output_side_packets=output_sequence_example=%s" % output_filename
    ]
    with open(input_filename, "wb") as input_file:
      input_file.write(sequence_example.SerializeToString())
    mediapipe_output = subprocess.check_output(cmd)
    if b"Failed to run the graph" in mediapipe_output:
      raise RuntimeError(mediapipe_output)
    with open(output_filename, "rb") as output_file:
      output_example = tf.train.SequenceExample()
      output_example.ParseFromString(output_file.read())
    os.close(input_fd)
    os.remove(input_filename)
    os.close(output_fd)
    os.remove(output_filename)
    return output_example


def bytes23(string):
  """Creates a bytes string in either Python 2 or  3."""
  if sys.version_info >= (3, 0):
    return bytes(string, "utf8")
  else:
    return bytes(string)


@contextlib.contextmanager
def _close_on_exit(writers):
  """Call close on all writers on exit."""
  try:
    yield writers
  finally:
    for writer in writers:
      writer.close()


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  DemoDataset(flags.FLAGS.path_to_demo_data).generate_examples(
      flags.FLAGS.path_to_mediapipe_binary, flags.FLAGS.path_to_graph_directory)


if __name__ == "__main__":
  flags.DEFINE_string("path_to_demo_data", "",
                      "Path to directory to write data to.")
  flags.DEFINE_string("path_to_mediapipe_binary", "",
                      "Path to the MediaPipe run_graph_file_io_main binary.")
  flags.DEFINE_string("path_to_graph_directory", "",
                      "Path to directory containing the graph files.")
  app.run(main)
