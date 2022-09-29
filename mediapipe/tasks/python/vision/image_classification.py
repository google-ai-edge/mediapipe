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
from typing import Callable, List, Mapping, Optional

from mediapipe.python import packet_creator
from mediapipe.python import packet_getter
from mediapipe.python._framework_bindings import image as image_module
from mediapipe.python._framework_bindings import packet as packet_module
from mediapipe.python._framework_bindings import task_runner as task_runner_module
from mediapipe.tasks.cc.vision.image_classifier.proto import image_classifier_options_pb2
from mediapipe.tasks.python.components.proto import classifier_options
from mediapipe.tasks.python.components.containers import classifications as classifications_module
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.core import task_info as task_info_module
from mediapipe.tasks.python.core.optional_dependencies import doc_controls
from mediapipe.tasks.python.vision.core import base_vision_task_api
from mediapipe.tasks.python.vision.core import vision_task_running_mode as running_mode_module

_BaseOptions = base_options_module.BaseOptions
_ImageClassifierOptionsProto = image_classifier_options_pb2.ImageClassifierOptions
_ClassifierOptions = classifier_options.ClassifierOptions
_RunningMode = running_mode_module.VisionTaskRunningMode
_TaskInfo = task_info_module.TaskInfo
_TaskRunner = task_runner_module.TaskRunner

_CLASSIFICATION_RESULT_OUT_STREAM_NAME = 'classification_result_out'
_CLASSIFICATION_RESULT_TAG = 'CLASSIFICATION_RESULT'
_IMAGE_IN_STREAM_NAME = 'image_in'
_IMAGE_TAG = 'IMAGE'
_TASK_GRAPH_NAME = 'mediapipe.tasks.vision.ImageClassifierGraph'


@dataclasses.dataclass
class ImageClassifierOptions:
  """Options for the image classifier task.

  Attributes:
    base_options: Base options for the image classifier task.
    running_mode: The running mode of the task. Default to the image mode.
      Image classifier task has three running modes:
      1) The image mode for classifying objects on single image inputs.
      2) The video mode for classifying objects on the decoded frames of a
         video.
      3) The live stream mode for classifying objects on a live stream of input
         data, such as from camera.
    classifier_options: Options for the image classification task.
    result_callback: The user-defined result callback for processing live stream
      data. The result callback should only be specified when the running mode
      is set to the live stream mode.
  """
  base_options: _BaseOptions
  running_mode: _RunningMode = _RunningMode.IMAGE
  classifier_options: _ClassifierOptions = _ClassifierOptions()
  result_callback: Optional[
      Callable[[classifications_module.ClassificationResult],
               None]] = None

  @doc_controls.do_not_generate_docs
  def to_pb2(self) -> _ImageClassifierOptionsProto:
    """Generates an ImageClassifierOptions protobuf object."""
    base_options_proto = self.base_options.to_pb2()
    base_options_proto.use_stream_mode = False if self.running_mode == _RunningMode.IMAGE else True
    classifier_options_proto = self.classifier_options.to_pb2()

    return _ImageClassifierOptionsProto(
        base_options=base_options_proto,
        classifier_options=classifier_options_proto
    )


class ImageClassifier(base_vision_task_api.BaseVisionTaskApi):
  """Class that performs image classification on images."""

  @classmethod
  def create_from_model_path(cls, model_path: str) -> 'ImageClassifier':
    """Creates an `ImageClassifier` object from a TensorFlow Lite model and the default `ImageClassifierOptions`.

    Note that the created `ImageClassifier` instance is in image mode, for
    detecting objects on single image inputs.

    Args:
      model_path: Path to the model.

    Returns:
      `ImageClassifier` object that's created from the model file and the default
      `ImageClassifierOptions`.

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

    def packets_callback(output_packets: Mapping[str, packet_module.Packet]):
      classification_result_proto = packet_getter.get_proto(
          output_packets[_CLASSIFICATION_RESULT_OUT_STREAM_NAME])

      classification_result = classifications_module.ClassificationResult([
          classifications_module.Classifications.create_from_pb2(classification)
          for classification in classification_result_proto.classifications
      ])
      options.result_callback(classification_result)

    task_info = _TaskInfo(
        task_graph=_TASK_GRAPH_NAME,
        input_streams=[':'.join([_IMAGE_TAG, _IMAGE_IN_STREAM_NAME])],
        output_streams=[
            ':'.join([_CLASSIFICATION_RESULT_TAG,
                      _CLASSIFICATION_RESULT_OUT_STREAM_NAME])
        ],
        task_options=options)
    return cls(
        task_info.generate_graph_config(
            enable_flow_limiting=options.running_mode ==
            _RunningMode.LIVE_STREAM), options.running_mode,
        packets_callback if options.result_callback else None)

  # TODO: Create an Image class for MediaPipe Tasks.
  def classify(
      self,
      image: image_module.Image
  ) -> classifications_module.ClassificationResult:
    """Performs image classification on the provided MediaPipe Image.

    Args:
      image: MediaPipe Image.

    Returns:
      A classification result object that contains a list of classifications.

    Raises:
      ValueError: If any of the input arguments is invalid.
      RuntimeError: If image classification failed to run.
    """
    output_packets = self._process_image_data(
        {_IMAGE_IN_STREAM_NAME: packet_creator.create_image(image)})
    classification_result_proto = packet_getter.get_proto(
        output_packets[_CLASSIFICATION_RESULT_OUT_STREAM_NAME])

    return classifications_module.ClassificationResult([
      classifications_module.Classifications.create_from_pb2(classification)
      for classification in classification_result_proto.classifications
    ])
