# Copyright 2022 The MediaPipe Authors. All Rights Reserved.
#
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
# ==============================================================================
"""Writes metadata and label file to the Text classifier models."""

from mediapipe.tasks.python.metadata.metadata_writers import metadata_writer

_MODEL_NAME = "TextClassifier"
_MODEL_DESCRIPTION = ("Classify the input text into a set of known categories.")


class MetadataWriter(metadata_writer.MetadataWriterBase):
  """MetadataWriter to write the metadata into the text classifier."""

  @classmethod
  def create_for_regex_model(
      cls, model_buffer: bytearray,
      regex_tokenizer: metadata_writer.RegexTokenizer,
      labels: metadata_writer.Labels) -> "MetadataWriter":
    """Creates MetadataWriter for TFLite model with regex tokentizer.

    The parameters required in this method are mandatory when using MediaPipe
    Tasks.

    Note that only the output TFLite is used for deployment. The output JSON
    content is used to interpret the metadata content.

    Args:
      model_buffer: A valid flatbuffer loaded from the TFLite model file.
      regex_tokenizer: information of the regex tokenizer [1] used to process
        the input string. If the tokenizer is `BertTokenizer` [2] or
        `SentencePieceTokenizer` [3], please refer to
        `create_for_bert_model`.
      labels: an instance of Labels helper class used in the output
        classification tensor [4].

      [1]:
        https://github.com/google/mediapipe/blob/f8af41b1eb49ff4bdad756ff19d1d36f486be614/mediapipe/tasks/metadata/metadata_schema.fbs#L500
      [2]:
        https://github.com/google/mediapipe/blob/f8af41b1eb49ff4bdad756ff19d1d36f486be614/mediapipe/tasks/metadata/metadata_schema.fbs#L477
      [3]:
        https://github.com/google/mediapipe/blob/f8af41b1eb49ff4bdad756ff19d1d36f486be614/mediapipe/tasks/metadata/metadata_schema.fbs#L485
      [4]:
        https://github.com/google/mediapipe/blob/f8af41b1eb49ff4bdad756ff19d1d36f486be614/mediapipe/tasks/metadata/metadata_schema.fbs#L99

    Returns:
      A MetadataWriter object.
    """
    writer = metadata_writer.MetadataWriter(model_buffer)
    writer.add_general_info(_MODEL_NAME, _MODEL_DESCRIPTION)
    writer.add_regex_text_input(regex_tokenizer)
    writer.add_classification_output(labels)
    return cls(writer)
