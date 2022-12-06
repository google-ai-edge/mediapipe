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
"""MediaPipe image classifier task."""

import dataclasses
from typing import Callable, Mapping, Optional, List

from mediapipe.python import packet_creator
from mediapipe.python import packet_getter
from mediapipe.python._framework_bindings import image as image_module
from mediapipe.python._framework_bindings import packet
from mediapipe.tasks.cc.components.containers.proto import classifications_pb2
from mediapipe.tasks.cc.components.processors.proto import classifier_options_pb2
from mediapipe.tasks.cc.vision.image_classifier.proto import image_classifier_graph_options_pb2
from mediapipe.tasks.python.components.containers import classification_result as classification_result_module
from mediapipe.tasks.python.components.containers import rect
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.core import task_info as task_info_module
from mediapipe.tasks.python.core.optional_dependencies import doc_controls
from mediapipe.tasks.python.vision.core import base_vision_task_api
from mediapipe.tasks.python.vision.core import image_processing_options as image_processing_options_module
from mediapipe.tasks.python.vision.core import vision_task_running_mode

ImageClassifierResult = classification_result_module.ClassificationResult
_NormalizedRect = rect.NormalizedRect
_BaseOptions = base_options_module.BaseOptions
_ImageClassifierGraphOptionsProto = image_classifier_graph_options_pb2.ImageClassifierGraphOptions
_ClassifierOptionsProto = classifier_options_pb2.ClassifierOptions
_RunningMode = vision_task_running_mode.VisionTaskRunningMode
_ImageProcessingOptions = image_processing_options_module.ImageProcessingOptions
_TaskInfo = task_info_module.TaskInfo

_CLASSIFICATIONS_STREAM_NAME = 'classifications_out'
_CLASSIFICATIONS_TAG = 'CLASSIFICATIONS'
_IMAGE_IN_STREAM_NAME = 'image_in'
_IMAGE_OUT_STREAM_NAME = 'image_out'
_IMAGE_TAG = 'IMAGE'
_NORM_RECT_STREAM_NAME = 'norm_rect_in'
_NORM_RECT_TAG = 'NORM_RECT'
_TASK_GRAPH_NAME = 'mediapipe.tasks.vision.image_classifier.ImageClassifierGraph'
_MICRO_SECONDS_PER_MILLISECOND = 1000


@dataclasses.dataclass
class ImageClassifierOptions:
  """Options for the image classifier task.

  Attributes:
    base_options: Base options for the image classifier task.
    running_mode: The running mode of the task. Default to the image mode. Image
      classifier task has three running modes: 1) The image mode for classifying
      objects on single image inputs. 2) The video mode for classifying objects
      on the decoded frames of a video. 3) The live stream mode for classifying
      objects on a live stream of input data, such as from camera.
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
    result_callback: The user-defined result callback for processing live stream
      data. The result callback should only be specified when the running mode
      is set to the live stream mode.
  """
  base_options: _BaseOptions
  running_mode: _RunningMode = _RunningMode.IMAGE
  display_names_locale: Optional[str] = None
  max_results: Optional[int] = None
  score_threshold: Optional[float] = None
  category_allowlist: Optional[List[str]] = None
  category_denylist: Optional[List[str]] = None
  result_callback: Optional[Callable[
      [ImageClassifierResult, image_module.Image, int], None]] = None

  @doc_controls.do_not_generate_docs
  def to_pb2(self) -> _ImageClassifierGraphOptionsProto:
    """Generates an ImageClassifierOptions protobuf object."""
    base_options_proto = self.base_options.to_pb2()
    base_options_proto.use_stream_mode = False if self.running_mode == _RunningMode.IMAGE else True
    classifier_options_proto = _ClassifierOptionsProto(
        score_threshold=self.score_threshold,
        category_allowlist=self.category_allowlist,
        category_denylist=self.category_denylist,
        display_names_locale=self.display_names_locale,
        max_results=self.max_results)

    return _ImageClassifierGraphOptionsProto(
        base_options=base_options_proto,
        classifier_options=classifier_options_proto)


class ImageClassifier(base_vision_task_api.BaseVisionTaskApi):
  """Class that performs image classification on images.

  The API expects a TFLite model with optional, but strongly recommended,
  TFLite Model Metadata.

  Input tensor:
    (kTfLiteUInt8/kTfLiteFloat32)
    - image input of size `[batch x height x width x channels]`.
    - batch inference is not supported (`batch` is required to be 1).
    - only RGB inputs are supported (`channels` is required to be 3).
    - if type is kTfLiteFloat32, NormalizationOptions are required to be
      attached to the metadata for input normalization.
  At least one output tensor with:
    (kTfLiteUInt8/kTfLiteFloat32)
    - `N `classes and either 2 or 4 dimensions, i.e. `[1 x N]` or
      `[1 x 1 x 1 x N]`
    - optional (but recommended) label map(s) as AssociatedFiles with type
      TENSOR_AXIS_LABELS, containing one label per line. The first such
      AssociatedFile (if any) is used to fill the `class_name` field of the
      results. The `display_name` field is filled from the AssociatedFile (if
      any) whose locale matches the `display_names_locale` field of the
      `ImageClassifierOptions` used at creation time ("en" by default, i.e.
      English). If none of these are available, only the `index` field of the
      results will be filled.
    - optional score calibration can be attached using ScoreCalibrationOptions
      and an AssociatedFile with type TENSOR_AXIS_SCORE_CALIBRATION. See
      metadata_schema.fbs [1] for more details.

  An example of such model can be found at:
  https://tfhub.dev/bohemian-visual-recognition-alliance/lite-model/models/mushroom-identification_v1/1

  [1]:
  https://github.com/google/mediapipe/blob/6cdc6443b6a7ed662744e2a2ce2d58d9c83e6d6f/mediapipe/tasks/metadata/metadata_schema.fbs#L456
  """

  @classmethod
  def create_from_model_path(cls, model_path: str) -> 'ImageClassifier':
    """Creates an `ImageClassifier` object from a TensorFlow Lite model and the default `ImageClassifierOptions`.

    Note that the created `ImageClassifier` instance is in image mode, for
    classifying objects on single image inputs.

    Args:
      model_path: Path to the model.

    Returns:
      `ImageClassifier` object that's created from the model file and the
      default `ImageClassifierOptions`.

    Raises:
      ValueError: If failed to create `ImageClassifier` object from the provided
        file such as invalid file path.
      RuntimeError: If other types of error occurred.
    """
    base_options = _BaseOptions(model_asset_path=model_path)
    options = ImageClassifierOptions(
        base_options=base_options, running_mode=_RunningMode.IMAGE)
    return cls.create_from_options(options)

  @classmethod
  def create_from_options(cls,
                          options: ImageClassifierOptions) -> 'ImageClassifier':
    """Creates the `ImageClassifier` object from image classifier options.

    Args:
      options: Options for the image classifier task.

    Returns:
      `ImageClassifier` object that's created from `options`.

    Raises:
      ValueError: If failed to create `ImageClassifier` object from
        `ImageClassifierOptions` such as missing the model.
      RuntimeError: If other types of error occurred.
    """

    def packets_callback(output_packets: Mapping[str, packet.Packet]):
      if output_packets[_IMAGE_OUT_STREAM_NAME].is_empty():
        return

      classification_result_proto = classifications_pb2.ClassificationResult()
      classification_result_proto.CopyFrom(
          packet_getter.get_proto(output_packets[_CLASSIFICATIONS_STREAM_NAME]))
      image = packet_getter.get_image(output_packets[_IMAGE_OUT_STREAM_NAME])
      timestamp = output_packets[_IMAGE_OUT_STREAM_NAME].timestamp
      options.result_callback(
          ImageClassifierResult.create_from_pb2(classification_result_proto),
          image, timestamp.value // _MICRO_SECONDS_PER_MILLISECOND)

    task_info = _TaskInfo(
        task_graph=_TASK_GRAPH_NAME,
        input_streams=[
            ':'.join([_IMAGE_TAG, _IMAGE_IN_STREAM_NAME]),
            ':'.join([_NORM_RECT_TAG, _NORM_RECT_STREAM_NAME]),
        ],
        output_streams=[
            ':'.join([_CLASSIFICATIONS_TAG, _CLASSIFICATIONS_STREAM_NAME]),
            ':'.join([_IMAGE_TAG, _IMAGE_OUT_STREAM_NAME])
        ],
        task_options=options)
    return cls(
        task_info.generate_graph_config(
            enable_flow_limiting=options.running_mode ==
            _RunningMode.LIVE_STREAM), options.running_mode,
        packets_callback if options.result_callback else None)

  def classify(
      self,
      image: image_module.Image,
      image_processing_options: Optional[_ImageProcessingOptions] = None
  ) -> ImageClassifierResult:
    """Performs image classification on the provided MediaPipe Image.

    Args:
      image: MediaPipe Image.
      image_processing_options: Options for image processing.

    Returns:
      A classification result object that contains a list of classifications.

    Raises:
      ValueError: If any of the input arguments is invalid.
      RuntimeError: If image classification failed to run.
    """
    normalized_rect = self.convert_to_normalized_rect(image_processing_options)
    output_packets = self._process_image_data({
        _IMAGE_IN_STREAM_NAME:
            packet_creator.create_image(image),
        _NORM_RECT_STREAM_NAME:
            packet_creator.create_proto(normalized_rect.to_pb2())
    })

    classification_result_proto = classifications_pb2.ClassificationResult()
    classification_result_proto.CopyFrom(
        packet_getter.get_proto(output_packets[_CLASSIFICATIONS_STREAM_NAME]))

    return ImageClassifierResult.create_from_pb2(classification_result_proto)

  def classify_for_video(
      self,
      image: image_module.Image,
      timestamp_ms: int,
      image_processing_options: Optional[_ImageProcessingOptions] = None
  ) -> ImageClassifierResult:
    """Performs image classification on the provided video frames.

    Only use this method when the ImageClassifier is created with the video
    running mode. It's required to provide the video frame's timestamp (in
    milliseconds) along with the video frame. The input timestamps should be
    monotonically increasing for adjacent calls of this method.

    Args:
      image: MediaPipe Image.
      timestamp_ms: The timestamp of the input video frame in milliseconds.
      image_processing_options: Options for image processing.

    Returns:
      A classification result object that contains a list of classifications.

    Raises:
      ValueError: If any of the input arguments is invalid.
      RuntimeError: If image classification failed to run.
    """
    normalized_rect = self.convert_to_normalized_rect(image_processing_options)
    output_packets = self._process_video_data({
        _IMAGE_IN_STREAM_NAME:
            packet_creator.create_image(image).at(
                timestamp_ms * _MICRO_SECONDS_PER_MILLISECOND),
        _NORM_RECT_STREAM_NAME:
            packet_creator.create_proto(normalized_rect.to_pb2()).at(
                timestamp_ms * _MICRO_SECONDS_PER_MILLISECOND)
    })

    classification_result_proto = classifications_pb2.ClassificationResult()
    classification_result_proto.CopyFrom(
        packet_getter.get_proto(output_packets[_CLASSIFICATIONS_STREAM_NAME]))

    return ImageClassifierResult.create_from_pb2(classification_result_proto)

  def classify_async(
      self,
      image: image_module.Image,
      timestamp_ms: int,
      image_processing_options: Optional[_ImageProcessingOptions] = None
  ) -> None:
    """Sends live image data (an Image with a unique timestamp) to perform image classification.

    Only use this method when the ImageClassifier is created with the live
    stream running mode. The input timestamps should be monotonically increasing
    for adjacent calls of this method. This method will return immediately after
    the input image is accepted. The results will be available via the
    `result_callback` provided in the `ImageClassifierOptions`. The
    `classify_async` method is designed to process live stream data such as
    camera input. To lower the overall latency, image classifier may drop the
    input images if needed. In other words, it's not guaranteed to have output
    per input image.

    The `result_callback` provides:
      - A classification result object that contains a list of classifications.
      - The input image that the image classifier runs on.
      - The input timestamp in milliseconds.

    Args:
      image: MediaPipe Image.
      timestamp_ms: The timestamp of the input image in milliseconds.
      image_processing_options: Options for image processing.

    Raises:
      ValueError: If the current input timestamp is smaller than what the image
        classifier has already processed.
    """
    normalized_rect = self.convert_to_normalized_rect(image_processing_options)
    self._send_live_stream_data({
        _IMAGE_IN_STREAM_NAME:
            packet_creator.create_image(image).at(
                timestamp_ms * _MICRO_SECONDS_PER_MILLISECOND),
        _NORM_RECT_STREAM_NAME:
            packet_creator.create_proto(normalized_rect.to_pb2()).at(
                timestamp_ms * _MICRO_SECONDS_PER_MILLISECOND)
    })
