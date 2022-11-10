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

from typing import Union

from mediapipe.tasks.python.metadata.metadata_writers import metadata_writer

_MODEL_NAME = "TextClassifier"
_MODEL_DESCRIPTION = ("Classify the input text into a set of known categories.")

# The input tensor names of models created by Model Maker.
_DEFAULT_ID_NAME = "serving_default_input_word_ids:0"
_DEFAULT_MASK_NAME = "serving_default_input_mask:0"
_DEFAULT_SEGMENT_ID_NAME = "serving_default_input_type_ids:0"


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

  @classmethod
  def create_for_bert_model(
      cls,
      model_buffer: bytearray,
      tokenizer: Union[metadata_writer.BertTokenizer,
                       metadata_writer.SentencePieceTokenizer],
      labels: metadata_writer.Labels,
      ids_name: str = _DEFAULT_ID_NAME,
      mask_name: str = _DEFAULT_MASK_NAME,
      segment_name: str = _DEFAULT_SEGMENT_ID_NAME,
  ) -> "MetadataWriter":
    """Creates MetadataWriter for models with {Bert/SentencePiece}Tokenizer.

    `ids_name`, `mask_name`, and `segment_name` correspond to the `Tensor.name`
    in the TFLite schema, which help to determine the tensor order when
    populating metadata. The default values come from Model Maker.

    Args:
      model_buffer: valid buffer of the model file.
      tokenizer: information of the tokenizer used to process the input string,
        if any. Supported tokenziers are: `BertTokenizer` [1] and
        `SentencePieceTokenizer` [2]. If the tokenizer is `RegexTokenizer` [3],
        refer to `create_for_regex_model`.
      labels: an instance of Labels helper class used in the output
        classification tensor [4].
      ids_name: name of the ids tensor, which represents the tokenized ids of
        the input text.
      mask_name: name of the mask tensor, which represents the mask with `1` for
        real tokens and `0` for padding tokens.
      segment_name: name of the segment ids tensor, where `0` stands for the
        first sequence, and `1` stands for the second sequence if exists. [1]:
        https://github.com/google/mediapipe/blob/f8af41b1eb49ff4bdad756ff19d1d36f486be614/mediapipe/tasks/metadata/metadata_schema.fbs#L477
          [2]:
        https://github.com/google/mediapipe/blob/f8af41b1eb49ff4bdad756ff19d1d36f486be614/mediapipe/tasks/metadata/metadata_schema.fbs#L485
          [3]:
        https://github.com/google/mediapipe/blob/f8af41b1eb49ff4bdad756ff19d1d36f486be614/mediapipe/tasks/metadata/metadata_schema.fbs#L500
          [4]:
        https://github.com/google/mediapipe/blob/f8af41b1eb49ff4bdad756ff19d1d36f486be614/mediapipe/tasks/metadata/metadata_schema.fbs#L99

    Returns:
      A MetadataWriter object.
    """
    writer = metadata_writer.MetadataWriter(model_buffer)
    writer.add_general_info(_MODEL_NAME, _MODEL_DESCRIPTION)
    writer.add_bert_text_input(tokenizer, ids_name, mask_name, segment_name)
    writer.add_classification_output(labels)
    return cls(writer)
