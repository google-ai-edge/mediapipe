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
"""MediaPipe text classifier task."""

import dataclasses

from mediapipe.python import packet_creator
from mediapipe.python import packet_getter
from mediapipe.tasks.cc.components.containers.proto import classifications_pb2
from mediapipe.tasks.cc.text.text_classifier.proto import text_classifier_graph_options_pb2
from mediapipe.tasks.python.components.containers import classifications
from mediapipe.tasks.python.components.processors import classifier_options
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.core import task_info as task_info_module
from mediapipe.tasks.python.core.optional_dependencies import doc_controls
from mediapipe.tasks.python.text.core import base_text_task_api

TextClassifierResult = classifications.ClassificationResult
_BaseOptions = base_options_module.BaseOptions
_TextClassifierGraphOptionsProto = text_classifier_graph_options_pb2.TextClassifierGraphOptions
_ClassifierOptions = classifier_options.ClassifierOptions
_TaskInfo = task_info_module.TaskInfo

_CLASSIFICATION_RESULT_OUT_STREAM_NAME = 'classification_result_out'
_CLASSIFICATION_RESULT_TAG = 'CLASSIFICATION_RESULT'
_TEXT_IN_STREAM_NAME = 'text_in'
_TEXT_TAG = 'TEXT'
_TASK_GRAPH_NAME = 'mediapipe.tasks.text.text_classifier.TextClassifierGraph'


@dataclasses.dataclass
class TextClassifierOptions:
  """Options for the text classifier task.

  Attributes:
    base_options: Base options for the text classifier task.
    classifier_options: Options for the text classification task.
  """
  base_options: _BaseOptions
  classifier_options: _ClassifierOptions = _ClassifierOptions()

  @doc_controls.do_not_generate_docs
  def to_pb2(self) -> _TextClassifierGraphOptionsProto:
    """Generates an TextClassifierOptions protobuf object."""
    base_options_proto = self.base_options.to_pb2()
    classifier_options_proto = self.classifier_options.to_pb2()

    return _TextClassifierGraphOptionsProto(
        base_options=base_options_proto,
        classifier_options=classifier_options_proto)


class TextClassifier(base_text_task_api.BaseTextTaskApi):
  """Class that performs classification on text."""

  @classmethod
  def create_from_model_path(cls, model_path: str) -> 'TextClassifier':
    """Creates an `TextClassifier` object from a TensorFlow Lite model and the default `TextClassifierOptions`.

    Args:
      model_path: Path to the model.

    Returns:
      `TextClassifier` object that's created from the model file and the
      default `TextClassifierOptions`.

    Raises:
      ValueError: If failed to create `TextClassifier` object from the provided
        file such as invalid file path.
      RuntimeError: If other types of error occurred.
    """
    base_options = _BaseOptions(model_asset_path=model_path)
    options = TextClassifierOptions(base_options=base_options)
    return cls.create_from_options(options)

  @classmethod
  def create_from_options(cls,
                          options: TextClassifierOptions) -> 'TextClassifier':
    """Creates the `TextClassifier` object from text classifier options.

    Args:
      options: Options for the text classifier task.

    Returns:
      `TextClassifier` object that's created from `options`.

    Raises:
      ValueError: If failed to create `TextClassifier` object from
        `TextClassifierOptions` such as missing the model.
      RuntimeError: If other types of error occurred.
    """
    task_info = _TaskInfo(
        task_graph=_TASK_GRAPH_NAME,
        input_streams=[':'.join([_TEXT_TAG, _TEXT_IN_STREAM_NAME])],
        output_streams=[
            ':'.join([
                _CLASSIFICATION_RESULT_TAG,
                _CLASSIFICATION_RESULT_OUT_STREAM_NAME
            ])
        ],
        task_options=options)
    return cls(task_info.generate_graph_config())

  def classify(self, text: str) -> TextClassifierResult:
    """Performs classification on the input `text`.

    Args:
      text: The input text.

    Returns:
      A `TextClassifierResult` object that contains a list of text
      classifications.

    Raises:
      ValueError: If any of the input arguments is invalid.
      RuntimeError: If text classification failed to run.
    """
    output_packets = self._runner.process(
        {_TEXT_IN_STREAM_NAME: packet_creator.create_string(text)})

    classification_result_proto = classifications_pb2.ClassificationResult()
    classification_result_proto.CopyFrom(
        packet_getter.get_proto(
            output_packets[_CLASSIFICATION_RESULT_OUT_STREAM_NAME]))

    return TextClassifierResult([
        classifications.Classifications.create_from_pb2(classification)
        for classification in classification_result_proto.classifications
    ])
