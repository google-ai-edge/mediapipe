# Copyright 2023 The MediaPipe Authors.
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
"""MediaPipe language detector task."""

import dataclasses
from typing import List, Optional

from mediapipe.python import packet_creator
from mediapipe.python import packet_getter
from mediapipe.tasks.cc.components.containers.proto import classifications_pb2
from mediapipe.tasks.cc.components.processors.proto import classifier_options_pb2
from mediapipe.tasks.cc.text.text_classifier.proto import text_classifier_graph_options_pb2
from mediapipe.tasks.python.components.containers import classification_result as classification_result_module
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.core import task_info as task_info_module
from mediapipe.tasks.python.core.optional_dependencies import doc_controls
from mediapipe.tasks.python.text.core import base_text_task_api

_ClassificationResult = classification_result_module.ClassificationResult
_BaseOptions = base_options_module.BaseOptions
_TextClassifierGraphOptionsProto = (
    text_classifier_graph_options_pb2.TextClassifierGraphOptions
)
_ClassifierOptionsProto = classifier_options_pb2.ClassifierOptions
_TaskInfo = task_info_module.TaskInfo

_CLASSIFICATIONS_STREAM_NAME = 'classifications_out'
_CLASSIFICATIONS_TAG = 'CLASSIFICATIONS'
_TEXT_IN_STREAM_NAME = 'text_in'
_TEXT_TAG = 'TEXT'
_TASK_GRAPH_NAME = 'mediapipe.tasks.text.text_classifier.TextClassifierGraph'


@dataclasses.dataclass
class LanguageDetectorResult:

  @dataclasses.dataclass
  class Detection:
    """A language code and its probability."""

    # An i18n language / locale code, e.g. "en" for English, "uz" for Uzbek,
    # "ja"-Latn for Japanese (romaji).
    language_code: str
    probability: float

  detections: List[Detection]


def _extract_language_detector_result(
    classification_result: classification_result_module.ClassificationResult,
) -> LanguageDetectorResult:
  """Extracts a LanguageDetectorResult from a ClassificationResult."""
  if len(classification_result.classifications) != 1:
    raise ValueError(
        'The LanguageDetector TextClassifierGraph should have exactly one '
        'classification head.'
    )
  languages_and_scores = classification_result.classifications[0]
  language_detector_result = LanguageDetectorResult([])
  for category in languages_and_scores.categories:
    if category.category_name is None:
      raise ValueError(
          'LanguageDetector ClassificationResult has a missing language code.'
      )
    prediction = LanguageDetectorResult.Detection(
        category.category_name, category.score
    )
    language_detector_result.detections.append(prediction)
  return language_detector_result


@dataclasses.dataclass
class LanguageDetectorOptions:
  """Options for the language detector task.

  Attributes:
    base_options: Base options for the language detector task.
    display_names_locale: The locale to use for display names specified through
      the TFLite Model Metadata.
    max_results: The maximum number of top-scored classification results to
      return.
    score_threshold: Overrides the ones provided in the model metadata. Results
      below this value are rejected.
    category_allowlist: Allowlist of category names. If non-empty,
      classification results whose category name is not in this set will be
      filtered out. Duplicate or unknown category names are ignored. Mutually
      exclusive with `category_denylist`.
    category_denylist: Denylist of category names. If non-empty, classification
      results whose category name is in this set will be filtered out. Duplicate
      or unknown category names are ignored. Mutually exclusive with
      `category_allowlist`.
  """

  base_options: _BaseOptions
  display_names_locale: Optional[str] = None
  max_results: Optional[int] = None
  score_threshold: Optional[float] = None
  category_allowlist: Optional[List[str]] = None
  category_denylist: Optional[List[str]] = None

  @doc_controls.do_not_generate_docs
  def to_pb2(self) -> _TextClassifierGraphOptionsProto:
    """Generates an TextClassifierOptions protobuf object."""
    base_options_proto = self.base_options.to_pb2()
    classifier_options_proto = _ClassifierOptionsProto(
        score_threshold=self.score_threshold,
        category_allowlist=self.category_allowlist,
        category_denylist=self.category_denylist,
        display_names_locale=self.display_names_locale,
        max_results=self.max_results,
    )

    return _TextClassifierGraphOptionsProto(
        base_options=base_options_proto,
        classifier_options=classifier_options_proto,
    )


class LanguageDetector(base_text_task_api.BaseTextTaskApi):
  """Class that predicts the language of an input text.

  This API expects a TFLite model with TFLite Model Metadata that contains the
  mandatory (described below) input tensors, output tensor, and the language
  codes in an AssociatedFile.

  Input tensors:
    (kTfLiteString)
    - 1 input tensor that is scalar or has shape [1] containing the input
      string.
  Output tensor:
    (kTfLiteFloat32)
    - 1 output tensor of shape`[1 x N]` where `N` is the number of languages.
  """

  @classmethod
  def create_from_model_path(cls, model_path: str) -> 'LanguageDetector':
    """Creates an `LanguageDetector` object from a TensorFlow Lite model and the default `LanguageDetectorOptions`.

    Args:
      model_path: Path to the model.

    Returns:
      `LanguageDetector` object that's created from the model file and the
      default `LanguageDetectorOptions`.

    Raises:
      ValueError: If failed to create `LanguageDetector` object from the
      provided
        file such as invalid file path.
      RuntimeError: If other types of error occurred.
    """
    base_options = _BaseOptions(model_asset_path=model_path)
    options = LanguageDetectorOptions(base_options=base_options)
    return cls.create_from_options(options)

  @classmethod
  def create_from_options(
      cls, options: LanguageDetectorOptions
  ) -> 'LanguageDetector':
    """Creates the `LanguageDetector` object from language detector options.

    Args:
      options: Options for the language detector task.

    Returns:
      `LanguageDetector` object that's created from `options`.

    Raises:
      ValueError: If failed to create `LanguageDetector` object from
        `LanguageDetectorOptions` such as missing the model.
      RuntimeError: If other types of error occurred.
    """
    task_info = _TaskInfo(
        task_graph=_TASK_GRAPH_NAME,
        input_streams=[':'.join([_TEXT_TAG, _TEXT_IN_STREAM_NAME])],
        output_streams=[
            ':'.join([_CLASSIFICATIONS_TAG, _CLASSIFICATIONS_STREAM_NAME])
        ],
        task_options=options,
    )
    return cls(task_info.generate_graph_config())

  def detect(self, text: str) -> LanguageDetectorResult:
    """Predicts the language of the input `text`.

    Args:
      text: The input text.

    Returns:
      A `LanguageDetectorResult` object that contains a list of languages and
      scores.

    Raises:
      ValueError: If any of the input arguments is invalid.
      RuntimeError: If language detection failed to run.
    """
    output_packets = self._runner.process(
        {_TEXT_IN_STREAM_NAME: packet_creator.create_string(text)}
    )

    classification_result_proto = classifications_pb2.ClassificationResult()
    classification_result_proto.CopyFrom(
        packet_getter.get_proto(output_packets[_CLASSIFICATIONS_STREAM_NAME])
    )

    classification_result = _ClassificationResult.create_from_pb2(
        classification_result_proto
    )
    return _extract_language_detector_result(classification_result)
