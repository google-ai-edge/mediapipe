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

r"""Code to download and parse the Kinetics dataset for TensorFlow models.

The [Kinetics data set](
https://deepmind.com/research/open-source/open-source-datasets/kinetics/)
is a data set for human action recognition maintained by DeepMind and Google.
This script downloads the annotations and prepares data from similar annotations
if local video files are available.

This script does not provide any means of accessing YouTube videos.

Running this code as a module generates the data set on disk. First, the
required files are downloaded (_download_data) which enables constructing the
label map. Then (in generate_examples), for each split in the data set, the
metadata is generated from the annotations for each example
(_generate_metadata), and MediaPipe is used to fill in the video frames
(_run_mediapipe). This script processes local video files defined in a custom
CSV in a comparable manner to the Kinetics data set for evaluating and
predicting values on your own data. The data set is written to disk as a set of
numbered TFRecord files.

The custom CSV format must match the Kinetics data set format, with columns
corresponding to [[label_name], video, start, end, split] followed by lines with
those fields. (Label_name is optional.) These field names can be used to
construct the paths to the video files using the Python string formatting
specification and the video_path_format_string flag:
   --video_path_format_string="/path/to/video/{video}.mp4"

Generating the data on disk can take considerable time and disk space.
(Image compression quality is the primary determiner of disk usage. TVL1 flow
determines runtime.)

Once the data is on disk, reading the data as a tf.data.Dataset is accomplished
with the following lines:

   kinetics = Kinetics("kinetics_data_path")
   dataset = kinetics.as_dataset("custom")
   # implement additional processing and batching here
   images_and_labels = dataset.make_one_shot_iterator().get_next()
   images = images_and_labels["images"]
   labels = image_and_labels["labels"]

This data is structured for per-clip action classification where images is
the sequence of images and labels are a one-hot encoded value. See
as_dataset() for more details.

Note that the number of videos changes in the data set over time, so it will
likely be necessary to change the expected number of examples.
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
import tarfile
import tempfile

from absl import app
from absl import flags
from absl import logging
from six.moves import range
from six.moves import urllib
from six.moves import zip
import tensorflow.compat.v1 as tf

from mediapipe.util.sequence import media_sequence as ms

CITATION = r"""@article{kay2017kinetics,
  title={The kinetics human action video dataset},
  author={Kay, Will and Carreira, Joao and Simonyan, Karen and Zhang, Brian and Hillier, Chloe and Vijayanarasimhan, Sudheendra and Viola, Fabio and Green, Tim and Back, Trevor and Natsev, Paul and others},
  journal={arXiv preprint arXiv:1705.06950},
  year={2017},
  url = {https://deepmind.com/research/open-source/kinetics},
}"""
ANNOTATION_URL = "https://storage.googleapis.com/deepmind-media/Datasets/kinetics700.tar.gz"
SECONDS_TO_MICROSECONDS = 1000000
GRAPHS = ["tvl1_flow_and_rgb_from_file.pbtxt"]
FILEPATTERN = "kinetics_700_%s_25fps_rgb_flow"
SPLITS = {
    "train": {
        "shards": 1000,
        "examples": 538779
    },
    "validate": {
        "shards": 100,
        "examples": 34499
    },
    "test": {
        "shards": 100,
        "examples": 68847
    },
    "custom": {
        "csv": None,  # Add a CSV for your own data here.
        "shards": 1,  # Change this number to increase sharding.
        "examples": -1
    },  # Negative 1 allows any number of examples.
}
NUM_CLASSES = 700


class Kinetics(object):
  """Generates and loads the Kinetics data set."""

  def __init__(self, path_to_data):
    if not path_to_data:
      raise ValueError("You must supply the path to the data directory.")
    self.path_to_data = path_to_data

  def as_dataset(self, split, shuffle=False, repeat=False,
                 serialized_prefetch_size=32, decoded_prefetch_size=32,
                 parse_labels=True):
    """Returns Kinetics as a tf.data.Dataset.

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
      parse_labels: if true, also returns the "labels" below. The only
        case where this should be false is if the data set was not constructed
        with a label map, resulting in this field being missing.
    Returns:
      A tf.data.Dataset object with the following structure: {
        "images": float tensor, shape [time, height, width, channels]
        "flow": float tensor, shape [time, height, width, 2]
        "num_frames": int32 tensor, shape [], number of frames in the sequence
        "labels": float32 tensor, shape [num_classes], one hot encoded. Only
          present if parse_labels is true.
    """
    logging.info("If you see an error about labels, and you don't supply "
                 "labels in your CSV, set parse_labels=False")
    def parse_fn(sequence_example):
      """Parses a Kinetics example."""
      context_features = {
          ms.get_example_id_key(): ms.get_example_id_default_parser(),
      }
      if parse_labels:
        context_features[
            ms.get_clip_label_string_key()] = tf.FixedLenFeature((), tf.string)
        context_features[
            ms.get_clip_label_index_key()] = tf.FixedLenFeature((), tf.int64)

      sequence_features = {
          ms.get_image_encoded_key(): ms.get_image_encoded_default_parser(),
          ms.get_forward_flow_encoded_key():
              ms.get_forward_flow_encoded_default_parser(),
      }
      parsed_context, parsed_sequence = tf.io.parse_single_sequence_example(
          sequence_example, context_features, sequence_features)

      images = tf.image.convert_image_dtype(
          tf.map_fn(tf.image.decode_jpeg,
                    parsed_sequence[ms.get_image_encoded_key()],
                    back_prop=False,
                    dtype=tf.uint8), tf.float32)
      num_frames = tf.shape(images)[0]

      flow = tf.image.convert_image_dtype(
          tf.map_fn(tf.image.decode_jpeg,
                    parsed_sequence[ms.get_forward_flow_encoded_key()],
                    back_prop=False,
                    dtype=tf.uint8), tf.float32)
      # The flow is quantized for storage in JPEGs by the FlowToImageCalculator.
      # The quantization needs to be inverted.
      flow = (flow[:, :, :, :2] - 0.5) * 2 * 20.

      output_dict = {
          "images": images,
          "flow": flow,
          "num_frames": num_frames,
      }
      if parse_labels:
        target = tf.one_hot(parsed_context[ms.get_clip_label_index_key()], 700)
        output_dict["labels"] = target
      return output_dict

    if split not in SPLITS:
      raise ValueError("Split %s not in %s" % split, str(list(SPLITS.keys())))
    all_shards = tf.io.gfile.glob(
        os.path.join(self.path_to_data, FILEPATTERN % split + "-*-of-*"))
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

  def generate_examples(self, path_to_mediapipe_binary,
                        path_to_graph_directory,
                        only_generate_metadata=False,
                        splits_to_process="train,val,test",
                        video_path_format_string=None,
                        download_labels_for_map=True):
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
      only_generate_metadata: If true, do not run mediapipe and write the
        metadata to disk instead.
      splits_to_process: csv string of which splits to process. Allows providing
        a custom CSV with the CSV flag. The original data is still downloaded
        to generate the label_map.
      video_path_format_string: The format string for the path to local files.
      download_labels_for_map: If true, download the annotations to create the
        label map.
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
    download_output = self._download_data(download_labels_for_map)
    for key in splits_to_process.split(","):
      logging.info("Generating metadata for split: %s", key)
      all_metadata = list(self._generate_metadata(
          key, download_output, video_path_format_string))
      logging.info("An example of the metadata: ")
      logging.info(all_metadata[0])
      random.seed(47)
      random.shuffle(all_metadata)
      shards = SPLITS[key]["shards"]
      shard_names = [os.path.join(
          self.path_to_data, FILEPATTERN % key + "-%05d-of-%05d" % (
              i, shards)) for i in range(shards)]
      writers = [tf.io.TFRecordWriter(shard_name) for shard_name in shard_names]
      with _close_on_exit(writers) as writers:
        for i, seq_ex in enumerate(all_metadata):
          if not only_generate_metadata:
            print("Processing example %d of %d   (%d%%) \r" % (
                i, len(all_metadata), i * 100 / len(all_metadata)), end="")
            for graph in GRAPHS:
              graph_path = os.path.join(path_to_graph_directory, graph)
              seq_ex = self._run_mediapipe(
                  path_to_mediapipe_binary, seq_ex, graph_path)
          writers[i % len(writers)].write(seq_ex.SerializeToString())
    logging.info("Data extraction complete.")

  def _generate_metadata(self, key, download_output,
                         video_path_format_string=None):
    """For each row in the annotation CSV, generates the corresponding metadata.

    Args:
      key: which split to process.
      download_output: the tuple output of _download_data containing
        - annotations_files: dict of keys to CSV annotation paths.
        - label_map: dict mapping from label strings to numeric indices.
      video_path_format_string: The format string for the path to local files.
    Yields:
      Each tf.SequenceExample of metadata, ready to pass to MediaPipe.
    """
    annotations_files, label_map = download_output
    with open(annotations_files[key], "r") as annotations:
      reader = csv.reader(annotations)
      for i, csv_row in enumerate(reader):
        if i == 0:  # the first row is the header
          continue
        # rename the row with a constitent set of names.
        if len(csv_row) == 5:
          row = dict(
              list(
                  zip(["label_name", "video", "start", "end", "split"],
                      csv_row)))
        else:
          row = dict(list(zip(["video", "start", "end", "split"], csv_row)))
        metadata = tf.train.SequenceExample()
        ms.set_example_id(bytes23(row["video"] + "_" + row["start"]),
                          metadata)
        ms.set_clip_media_id(bytes23(row["video"]), metadata)
        ms.set_clip_alternative_media_id(bytes23(row["split"]), metadata)
        if video_path_format_string:
          filepath = video_path_format_string.format(**row)
          ms.set_clip_data_path(bytes23(filepath), metadata)
        assert row["start"].isdigit(), "Invalid row: %s" % str(row)
        assert row["end"].isdigit(), "Invalid row: %s" % str(row)
        if "label_name" in row:
          ms.set_clip_label_string([bytes23(row["label_name"])], metadata)
          if label_map:
            ms.set_clip_label_index([label_map[row["label_name"]]], metadata)
        yield metadata

  def _download_data(self, download_labels_for_map):
    """Downloads and extracts data if not already available."""
    if sys.version_info >= (3, 0):
      urlretrieve = urllib.request.urlretrieve
    else:
      urlretrieve = urllib.request.urlretrieve
    logging.info("Creating data directory.")
    tf.io.gfile.makedirs(self.path_to_data)
    logging.info("Downloading annotations.")
    paths = {}
    if download_labels_for_map:
      tar_path = os.path.join(self.path_to_data, ANNOTATION_URL.split("/")[-1])
      if not tf.io.gfile.exists(tar_path):
        urlretrieve(ANNOTATION_URL, tar_path)
        with tarfile.open(tar_path) as annotations_tar:
          def is_within_directory(directory, target):
              
              abs_directory = os.path.abspath(directory)
              abs_target = os.path.abspath(target)
          
              prefix = os.path.commonprefix([abs_directory, abs_target])
              
              return prefix == abs_directory
          
          def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
          
              for member in tar.getmembers():
                  member_path = os.path.join(path, member.name)
                  if not is_within_directory(path, member_path):
                      raise Exception("Attempted Path Traversal in Tar File")
          
              tar.extractall(path, members, numeric_owner=numeric_owner) 
              
          
          safe_extract(annotations_tar, self.path_to_data)
      for split in ["train", "test", "validate"]:
        csv_path = os.path.join(self.path_to_data, "kinetics700/%s.csv" % split)
        if not tf.io.gfile.exists(csv_path):
          with tarfile.open(tar_path) as annotations_tar:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(annotations_tar, self.path_to_data)
        paths[split] = csv_path
    for split, contents in SPLITS.items():
      if "csv" in contents and contents["csv"]:
        paths[split] = contents["csv"]
    label_map = (self.get_label_map_and_verify_example_counts(paths) if
                 download_labels_for_map else None)
    return paths, label_map

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

  def get_label_map_and_verify_example_counts(self, paths):
    """Verify the number of examples and labels have not changed."""
    for name, path in paths.items():
      with open(path, "r") as f:
        lines = f.readlines()
      # the header adds one line and one "key".
      num_examples = len(lines) - 1
      keys = [l.split(",")[0] for l in lines]
      label_map = None
      if name == "train":
        classes = sorted(list(set(keys[1:])))
        num_keys = len(set(keys)) - 1
        assert NUM_CLASSES == num_keys, (
            "Found %d labels for split: %s, should be %d" % (
                num_keys, name, NUM_CLASSES))
        label_map = dict(list(zip(classes, list(range(len(classes))))))
      if SPLITS[name]["examples"] > 0:
        assert SPLITS[name]["examples"] == num_examples, (
            "Found %d examples for split: %s, should be %d" % (
                num_examples, name, SPLITS[name]["examples"]))
    return label_map


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
  if flags.FLAGS.path_to_custom_csv:
    SPLITS["custom"]["csv"] = flags.FLAGS.path_to_custom_csv
  Kinetics(flags.FLAGS.path_to_kinetics_data).generate_examples(
      flags.FLAGS.path_to_mediapipe_binary,
      flags.FLAGS.path_to_graph_directory,
      flags.FLAGS.only_generate_metadata,
      flags.FLAGS.splits_to_process,
      flags.FLAGS.video_path_format_string,
      flags.FLAGS.download_labels_for_map)

if __name__ == "__main__":
  flags.DEFINE_string("path_to_kinetics_data",
                      "",
                      "Path to directory to write data to.")
  flags.DEFINE_string("path_to_mediapipe_binary",
                      "",
                      "Path to the MediaPipe run_graph_file_io_main binary.")
  flags.DEFINE_string("path_to_graph_directory",
                      "",
                      "Path to directory containing the graph files.")
  flags.DEFINE_boolean("only_generate_metadata",
                       False,
                       "If true, only generate the metadata files.")
  flags.DEFINE_boolean("download_labels_for_map",
                       True,
                       "If true, download the annotations to construct the "
                       "label map.")
  flags.DEFINE_string("splits_to_process",
                      "custom",
                      "Process these splits. Useful for custom data splits.")
  flags.DEFINE_string("video_path_format_string",
                      None,
                      "The format string for the path to local video files. "
                      "Uses the Python string.format() syntax with possible "
                      "arguments of {video}, {start}, {end}, {label_name}, and "
                      "{split}, corresponding to columns of the data csvs.")
  flags.DEFINE_string("path_to_custom_csv",
                      None,
                      "If present, processes this CSV as a custom split.")
  app.run(main)
