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

r"""Code to download and parse the Charades dataset for TensorFlow models.

The [Charades data set](https://allenai.org/plato/charades/) is a data set of
human action recognition collected with and maintained by the Allen Institute
for Artificial Intelligence. This script downloads and prepares the data set for
training a TensorFlow model. To use this script, you must abide by the
[lincense](https://allenai.org/plato/charades/license.txt) for the Charades data
set provided by the Allen Institute. The license for this script only covers
this code and not the data set.

Running this code as a module generates the data set on disk. First, the
required files are downloaded (_download_data). Then, for each split in the
data set (generate_examples), the metadata is generated from the annotations for
each example (_generate_metadata), and MediaPipe is used to fill in the video
frames (_run_mediapipe). The data set is written to disk as a set of numbered
TFRecord files. If the download is disrupted, the incomplete files will need to
be removed before running the script again. This pattern can be reproduced and
modified to generate most video data sets.

Generating the data on disk will probably take 4-8 hours and requires 150 GB of
disk space. (Image compression quality is the primary determiner of disk usage.)
After generating the data, the 30 GB of compressed video data can be deleted.

Once the data is on disk, reading the data as a tf.data.Dataset is accomplished
with the following lines:

   charades = CharadesDataset("charades_data_path")
   dataset = charades.as_dataset("test")
   # implement additional processing and batching here
   images_and_labels = dataset.make_one_shot_iterator().get_next()
   images = images_and_labels["images"]
   labels = image_and_labels["classification_target"]
   label_weights = image_and_labels["indicator_matrix"]

This data is structured for per-frame action classification where images is
the sequence of images, labels are the sequence of classification targets and,
label_weights is 1 for valid frames and 0 for padded frames (if any). See
as_dataset() for more details.
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
import zipfile

from absl import app
from absl import flags
from absl import logging
from six.moves import range
from six.moves import urllib
import tensorflow.compat.v1 as tf

from mediapipe.util.sequence import media_sequence as ms


DATA_URL_ANNOTATIONS = "http://ai2-website.s3.amazonaws.com/data/Charades.zip"
DATA_URL_VIDEOS = "http://ai2-website.s3.amazonaws.com/data/Charades_v1_480.zip"
DATA_URL_LICENSE = "https://allenai.org/plato/charades/license.txt"
CITATION = r"""@article{sigurdsson2016hollywood,
author = {Gunnar A. Sigurdsson and G{\"u}l Varol and Xiaolong Wang and Ivan Laptev and Ali Farhadi and Abhinav Gupta},
title = {Hollywood in Homes: Crowdsourcing Data Collection for Activity Understanding},
journal = {ArXiv e-prints},
eprint = {1604.01753},
year = {2016},
url = {http://arxiv.org/abs/1604.01753},
}"""
SECONDS_TO_MICROSECONDS = 1000000
GRAPHS = ["clipped_images_from_file_at_24fps.pbtxt"]
SPLITS = {
    "train": ("charades_v1_train_records",  # base name for sharded files
              "Charades_v1_train.csv",  # path to csv of annotations
              1000,  # number of shards
              7986),  # number of examples
    "test": ("charades_v1_test_records",
             "Charades_v1_test.csv",
             100,
             1864),
}
NUM_CLASSES = 157
CLASS_LABEL_OFFSET = 1


class Charades(object):
  """Generates and loads the Charades data set."""

  def __init__(self, path_to_data):
    if not path_to_data:
      raise ValueError("You must supply the path to the data directory.")
    self.path_to_data = path_to_data

  def as_dataset(self, split, shuffle=False, repeat=False,
                 serialized_prefetch_size=32, decoded_prefetch_size=32):
    """Returns Charades as a tf.data.Dataset.

    After running this function, calling padded_batch() on the Dataset object
    will produce batches of data, but additional preprocessing may be desired.
    If using padded_batch, the indicator_matrix output distinguishes valid
    from padded frames.

    Args:
      split: either "train" or "test"
      shuffle: if true, shuffles both files and examples.
      repeat: if true, repeats the data set forever.
      serialized_prefetch_size: the buffer size for reading from disk.
      decoded_prefetch_size: the buffer size after decoding.
    Returns:
      A tf.data.Dataset object with the following structure: {
        "images": uint8 tensor, shape [time, height, width, channels]
        "segment_matrix": binary tensor of segments, shape [time, num_segments].
          See one_hot_segments() for details.
        "indicator_matrix": binary tensor indicating valid frames,
          shape [time, 1]. If padded with zeros to align sizes, the indicator
          marks where segments is valid.
        "classification_target": binary tensor of classification targets,
          shape [time, 158 classes]. More than one value in a row can be 1.0 if
          segments overlap.
        "example_id": a unique string id for each example, shape [].
        "sampling_rate": the frame rate for each sequence, shape [].
        "gt_segment_seconds": the start and end time of each segment,
          shape [num_segments, 2].
        "gt_segment_classes": the class labels for each segment,
          shape [num_segments].
        "num_segments": the number of segments in the example, shape [].
        "num_timesteps": the number of timesteps in the example, shape [].
    """
    def parse_fn(sequence_example):
      """Parses a Charades example."""
      context_features = {
          ms.get_example_id_key(): ms.get_example_id_default_parser(),
          ms.get_segment_start_index_key(): (
              ms.get_segment_start_index_default_parser()),
          ms.get_segment_end_index_key(): (
              ms.get_segment_end_index_default_parser()),
          ms.get_segment_label_index_key(): (
              ms.get_segment_label_index_default_parser()),
          ms.get_segment_label_string_key(): (
              ms.get_segment_label_string_default_parser()),
          ms.get_segment_start_timestamp_key(): (
              ms.get_segment_start_timestamp_default_parser()),
          ms.get_segment_end_timestamp_key(): (
              ms.get_segment_end_timestamp_default_parser()),
          ms.get_image_frame_rate_key(): (
              ms.get_image_frame_rate_default_parser()),
      }

      sequence_features = {
          ms.get_image_encoded_key(): ms.get_image_encoded_default_parser()
      }
      parsed_context, parsed_sequence = tf.io.parse_single_sequence_example(
          sequence_example, context_features, sequence_features)

      sequence_length = tf.shape(parsed_sequence[ms.get_image_encoded_key()])[0]
      num_segments = tf.shape(
          parsed_context[ms.get_segment_label_index_key()])[0]
      # segments matrix and targets for training.
      segments_matrix, indicator = one_hot_segments(
          tf.sparse_tensor_to_dense(
              parsed_context[ms.get_segment_start_index_key()]),
          tf.sparse_tensor_to_dense(
              parsed_context[ms.get_segment_end_index_key()]),
          sequence_length)

      classification_target = timepoint_classification_target(
          segments_matrix,
          tf.sparse_tensor_to_dense(
              parsed_context[ms.get_segment_label_index_key()]
              ) + CLASS_LABEL_OFFSET,
          NUM_CLASSES + CLASS_LABEL_OFFSET)

      # [segments, 2] start and end time in seconds.
      gt_segment_seconds = tf.to_float(tf.concat(
          [tf.expand_dims(tf.sparse_tensor_to_dense(parsed_context[
              ms.get_segment_start_timestamp_key()]), 1),
           tf.expand_dims(tf.sparse_tensor_to_dense(parsed_context[
               ms.get_segment_end_timestamp_key()]), 1)],
          1)) / float(SECONDS_TO_MICROSECONDS)
      gt_segment_classes = tf.sparse_tensor_to_dense(parsed_context[
          ms.get_segment_label_index_key()]) + CLASS_LABEL_OFFSET
      example_id = parsed_context[ms.get_example_id_key()]
      sampling_rate = parsed_context[ms.get_image_frame_rate_key()]

      images = tf.map_fn(tf.image.decode_jpeg,
                         parsed_sequence[ms.get_image_encoded_key()],
                         back_prop=False,
                         dtype=tf.uint8)

      output_dict = {
          "segment_matrix": segments_matrix,
          "indicator_matrix": indicator,
          "classification_target": classification_target,
          "example_id": example_id,
          "sampling_rate": sampling_rate,
          "gt_segment_seconds": gt_segment_seconds,
          "gt_segment_classes": gt_segment_classes,
          "num_segments": num_segments,
          "num_timesteps": sequence_length,
          "images": images,
      }
      return output_dict

    if split not in SPLITS:
      raise ValueError("Split %s not in %s" % split, str(list(SPLITS.keys())))
    all_shards = tf.io.gfile.glob(
        os.path.join(self.path_to_data, SPLITS[split][0] + "-*-of-*"))
    random.shuffle(all_shards)
    all_shards_dataset = tf.data.Dataset.from_tensor_slices(all_shards)
    cycle_length = min(16, len(all_shards))
    dataset = all_shards_dataset.apply(
        tf.contrib.data.parallel_interleave(
            tf.data.TFRecordDataset,
            cycle_length=cycle_length,
            block_length=1, sloppy=True,
            buffer_output_elements=serialized_prefetch_size))
    dataset = dataset.prefetch(serialized_prefetch_size)
    if shuffle:
      dataset = dataset.shuffle(serialized_prefetch_size)
    if repeat:
      dataset = dataset.repeat()
    dataset = dataset.map(parse_fn)
    dataset = dataset.prefetch(decoded_prefetch_size)
    return dataset

  def generate_examples(self,
                        path_to_mediapipe_binary, path_to_graph_directory):
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
      raise ValueError(
          "You must supply the path to the MediaPipe binary for "
          "mediapipe/examples/desktop/demo:media_sequence_demo.")
    if not path_to_graph_directory:
      raise ValueError(
          "You must supply the path to the directory with MediaPipe graphs in "
          "mediapipe/graphs/media_sequence/.")
    logging.info("Downloading data.")
    annotation_dir, video_dir = self._download_data()
    for name, annotations, shards, _ in SPLITS.values():
      annotation_file = os.path.join(
          annotation_dir, annotations)
      logging.info("Generating metadata for split: %s", name)
      all_metadata = list(self._generate_metadata(annotation_file, video_dir))
      random.seed(47)
      random.shuffle(all_metadata)
      shard_names = [os.path.join(self.path_to_data, name + "-%05d-of-%05d" % (
          i, shards)) for i in range(shards)]
      writers = [tf.io.TFRecordWriter(shard_name) for shard_name in shard_names]
      with _close_on_exit(writers) as writers:
        for i, seq_ex in enumerate(all_metadata):
          print("Processing example %d of %d   (%d%%) \r" % (
              i, len(all_metadata), i * 100 / len(all_metadata)), end="")
          for graph in GRAPHS:
            graph_path = os.path.join(path_to_graph_directory, graph)
            seq_ex = self._run_mediapipe(
                path_to_mediapipe_binary, seq_ex, graph_path)
          writers[i % len(writers)].write(seq_ex.SerializeToString())
    logging.info("Data extraction complete.")

  def _generate_metadata(self, annotations_file, video_dir):
    """For each row in the annotation CSV, generates the corresponding metadata.

    Args:
      annotations_file: path to the file of Charades CSV annotations.
      video_dir: path to the directory of video files referenced by the
        annotations.
    Yields:
      Each tf.SequenceExample of metadata, ready to pass to MediaPipe.
    """
    with open(annotations_file, "r") as annotations:
      reader = csv.DictReader(annotations)
      for row in reader:
        metadata = tf.train.SequenceExample()
        filepath = os.path.join(video_dir, "%s.mp4" % row["id"])
        actions = row["actions"].split(";")
        action_indices = []
        action_strings = []
        action_start_times = []
        action_end_times = []
        for action in actions:
          if not action:
            continue
          string, start, end = action.split(" ")
          action_indices.append(int(string[1:]))
          action_strings.append(bytes23(string))
          action_start_times.append(int(float(start) * SECONDS_TO_MICROSECONDS))
          action_end_times.append(int(float(end) * SECONDS_TO_MICROSECONDS))
        ms.set_example_id(bytes23(row["id"]), metadata)
        ms.set_clip_data_path(bytes23(filepath), metadata)
        ms.set_clip_start_timestamp(0, metadata)
        ms.set_clip_end_timestamp(
            int(float(row["length"]) * SECONDS_TO_MICROSECONDS), metadata)
        ms.set_segment_start_timestamp(action_start_times, metadata)
        ms.set_segment_end_timestamp(action_end_times, metadata)
        ms.set_segment_label_string(action_strings, metadata)
        ms.set_segment_label_index(action_indices, metadata)
        yield metadata

  def _download_data(self):
    """Downloads and extracts data if not already available."""
    if sys.version_info >= (3, 0):
      urlretrieve = urllib.request.urlretrieve
    else:
      urlretrieve = urllib.request.urlretrieve
    logging.info("Creating data directory.")
    tf.io.gfile.makedirs(self.path_to_data)
    logging.info("Downloading license.")
    local_license_path = os.path.join(
        self.path_to_data, DATA_URL_LICENSE.split("/")[-1])
    if not tf.io.gfile.exists(local_license_path):
      urlretrieve(DATA_URL_LICENSE, local_license_path)
    logging.info("Downloading annotations.")
    local_annotations_path = os.path.join(
        self.path_to_data, DATA_URL_ANNOTATIONS.split("/")[-1])
    if not tf.io.gfile.exists(local_annotations_path):
      urlretrieve(DATA_URL_ANNOTATIONS, local_annotations_path)
    logging.info("Downloading videos.")
    local_videos_path = os.path.join(
        self.path_to_data, DATA_URL_VIDEOS.split("/")[-1])
    if not tf.io.gfile.exists(local_videos_path):
      urlretrieve(DATA_URL_VIDEOS, local_videos_path, progress_hook)
    logging.info("Extracting annotations.")
    # return video dir and annotation_dir by removing .zip from the path.
    annotations_dir = local_annotations_path[:-4]
    if not tf.io.gfile.exists(annotations_dir):
      with zipfile.ZipFile(local_annotations_path) as annotations_zip:
        annotations_zip.extractall(self.path_to_data)
    logging.info("Extracting videos.")
    video_dir = local_videos_path[:-4]
    if not tf.io.gfile.exists(video_dir):
      with zipfile.ZipFile(local_videos_path) as videos_zip:
        videos_zip.extractall(self.path_to_data)
    return annotations_dir, video_dir

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
    cmd = [path_to_mediapipe_binary,
           "--calculator_graph_config_file=%s" % graph,
           "--input_side_packets=input_sequence_example=%s" % input_filename,
           "--output_side_packets=output_sequence_example=%s" % output_filename]
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


def one_hot_segments(start_indices, end_indices, num_samples):
  """Returns a one-hot, float matrix of segments at each timestep.

  All integers in the inclusive range of start_indices and end_indices are used.
  This allows start and end timestamps to be mapped to the same index and the
  segment will not be omitted.

  Args:
    start_indices: a 1d tensor of integer indices for the start of each
      segement.
    end_indices: a tensor of integer indices for the end of each segment.
      Must be the same shape as start_indices. Values should be >= start_indices
      but not strictly enforced.
    num_samples: the number of rows in the output. Indices should be <
      num_samples, but this is not strictly enforced.
  Returns:
    (segments, indicator)
    segments: A [num_samples, num_elements(start_indices)] tensor where in each
      column the rows with indices >= start_indices[column] and
      <= end_indices[column] are 1.0 and all other values are 0.0.
    indicator: a tensor of 1.0 values with shape [num_samples, 1]. If padded
      with zeros to align sizes, the indicator marks where segments is valid.
  """
  start_indices = tf.convert_to_tensor(start_indices)
  end_indices = tf.convert_to_tensor(end_indices)
  start_indices.shape.assert_is_compatible_with(end_indices.shape)
  start_indices.shape.assert_has_rank(1)
  end_indices.shape.assert_has_rank(1)
  # create a matrix of the index at each row with a column per segment.
  indices = tf.to_int64(
      tf.tile(
          tf.transpose(tf.expand_dims(tf.range(num_samples), 0)),
          [1, tf.shape(start_indices)[0]]))
  # switch to one hot encoding of segments (includes start and end indices)
  segments = tf.to_float(
      tf.logical_and(
          tf.greater_equal(indices, start_indices),
          tf.less_equal(indices, end_indices)))
  # create a tensors of ones everywhere there's an annotation. If padded with
  # zeros later, element-wise multiplication of the loss will mask out the
  # padding.
  indicator = tf.ones(shape=[num_samples, 1], dtype=tf.float32)
  return segments, indicator


def timepoint_classification_target(segments, segment_classes, num_classes):
  """Produces a classification target at each timepoint.

  If no segments are present at a time point, the first class is set to 1.0.
  This should be used as a background class unless segments are always present.

  Args:
    segments: a [time, num_segments] tensor that is 1.0 at indices within
      each segment and 0.0 elsewhere.
    segment_classes: a [num_segments] tensor with the class index of each
      segment.
    num_classes: the number of classes (must be >= max(segment_classes) + 1)
  Returns:
    a [time, num_classes] tensor. In the final output, more than one
    value in a row can be 1.0 if segments overlap.
  """
  num_segments = tf.shape(segments)[1]
  matrix_of_class_indices = tf.to_int32(
      segments * tf.to_float(tf.expand_dims(segment_classes, 0)))
  # First column will have one count per zero segment. Correct this to be 0
  # unless no segments are present.
  one_hot = tf.reduce_sum(tf.one_hot(matrix_of_class_indices, num_classes), 1)
  normalizer = tf.concat([
      tf.ones(shape=[1, 1], dtype=tf.float32) / tf.to_float(num_segments),
      tf.ones(shape=[1, num_classes - 1], dtype=tf.float32)
  ], 1)
  corrected_one_hot = tf.floor(one_hot * normalizer)
  return corrected_one_hot


def progress_hook(blocks, block_size, total_size):
  print("Downloaded %d%% of %d bytes   (%d blocks)\r" % (
      blocks * block_size / total_size * 100, total_size, blocks), end="")


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
  Charades(flags.FLAGS.path_to_charades_data).generate_examples(
      flags.FLAGS.path_to_mediapipe_binary,
      flags.FLAGS.path_to_graph_directory)

if __name__ == "__main__":
  flags.DEFINE_string("path_to_charades_data",
                      "",
                      "Path to directory to write data to.")
  flags.DEFINE_string("path_to_mediapipe_binary",
                      "",
                      "Path to the MediaPipe run_graph_file_io_main binary.")
  flags.DEFINE_string("path_to_graph_directory",
                      "",
                      "Path to directory containing the graph files.")
  app.run(main)
