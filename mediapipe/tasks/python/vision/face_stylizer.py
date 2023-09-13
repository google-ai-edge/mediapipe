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
"""MediaPipe face stylizer task."""

import dataclasses
from typing import Optional

from mediapipe.python import packet_creator
from mediapipe.python import packet_getter
from mediapipe.python._framework_bindings import image as image_module
from mediapipe.tasks.cc.vision.face_stylizer.proto import face_stylizer_graph_options_pb2
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.core import task_info as task_info_module
from mediapipe.tasks.python.core.optional_dependencies import doc_controls
from mediapipe.tasks.python.vision.core import base_vision_task_api
from mediapipe.tasks.python.vision.core import image_processing_options as image_processing_options_module
from mediapipe.tasks.python.vision.core import vision_task_running_mode as running_mode_module

_BaseOptions = base_options_module.BaseOptions
_FaceStylizerGraphOptionsProto = (
    face_stylizer_graph_options_pb2.FaceStylizerGraphOptions
)
_ImageProcessingOptions = image_processing_options_module.ImageProcessingOptions
_TaskInfo = task_info_module.TaskInfo

_STYLIZED_IMAGE_NAME = 'stylized_image'
_STYLIZED_IMAGE_TAG = 'STYLIZED_IMAGE'
_NORM_RECT_STREAM_NAME = 'norm_rect_in'
_NORM_RECT_TAG = 'NORM_RECT'
_IMAGE_IN_STREAM_NAME = 'image_in'
_IMAGE_OUT_STREAM_NAME = 'image_out'
_IMAGE_TAG = 'IMAGE'
_TASK_GRAPH_NAME = 'mediapipe.tasks.vision.face_stylizer.FaceStylizerGraph'
_MICRO_SECONDS_PER_MILLISECOND = 1000


@dataclasses.dataclass
class FaceStylizerOptions:
  """Options for the face stylizer task.

  Attributes:
    base_options: Base options for the face stylizer task.
  """

  base_options: _BaseOptions

  @doc_controls.do_not_generate_docs
  def to_pb2(self) -> _FaceStylizerGraphOptionsProto:
    """Generates an FaceStylizerOptions protobuf object."""
    base_options_proto = self.base_options.to_pb2()
    return _FaceStylizerGraphOptionsProto(base_options=base_options_proto)


class FaceStylizer(base_vision_task_api.BaseVisionTaskApi):
  """Class that performs face stylization on images."""

  @classmethod
  def create_from_model_path(cls, model_path: str) -> 'FaceStylizer':
    """Creates an `FaceStylizer` object from a TensorFlow Lite model and the default `FaceStylizerOptions`.

    Note that the created `FaceStylizer` instance is in image mode, for
    stylizing one face on a single image input.

    Args:
      model_path: Path to the model.

    Returns:
      `FaceStylizer` object that's created from the model file and the default
      `FaceStylizerOptions`.

    Raises:
      ValueError: If failed to create `FaceStylizer` object from the provided
        file such as invalid file path.
      RuntimeError: If other types of error occurred.
    """
    base_options = _BaseOptions(model_asset_path=model_path)
    options = FaceStylizerOptions(base_options=base_options)
    return cls.create_from_options(options)

  @classmethod
  def create_from_options(cls, options: FaceStylizerOptions) -> 'FaceStylizer':
    """Creates the `FaceStylizer` object from face stylizer options.

    Args:
      options: Options for the face stylizer task.

    Returns:
      `FaceStylizer` object that's created from `options`.

    Raises:
      ValueError: If failed to create `FaceStylizer` object from
        `FaceStylizerOptions` such as missing the model.
      RuntimeError: If other types of error occurred.
    """

    task_info = _TaskInfo(
        task_graph=_TASK_GRAPH_NAME,
        input_streams=[
            ':'.join([_IMAGE_TAG, _IMAGE_IN_STREAM_NAME]),
            ':'.join([_NORM_RECT_TAG, _NORM_RECT_STREAM_NAME]),
        ],
        output_streams=[
            ':'.join([_STYLIZED_IMAGE_TAG, _STYLIZED_IMAGE_NAME]),
            ':'.join([_IMAGE_TAG, _IMAGE_OUT_STREAM_NAME]),
        ],
        task_options=options,
    )
    return cls(
        task_info.generate_graph_config(),
        running_mode=running_mode_module.VisionTaskRunningMode.IMAGE,
    )

  def stylize(
      self,
      image: image_module.Image,
      image_processing_options: Optional[_ImageProcessingOptions] = None,
  ) -> image_module.Image:
    """Performs face stylization on the provided MediaPipe Image.

    Only use this method when the FaceStylizer is created with the image
    running mode.

    Args:
      image: MediaPipe Image.
      image_processing_options: Options for image processing.

    Returns:
      The stylized image of the most visible face. The stylized output image
      size is the same as the model output size. None if no face is detected
      on the input image.

    Raises:
      ValueError: If any of the input arguments is invalid.
      RuntimeError: If face stylization failed to run.
    """
    normalized_rect = self.convert_to_normalized_rect(
        image_processing_options, image
    )
    output_packets = self._process_image_data({
        _IMAGE_IN_STREAM_NAME: packet_creator.create_image(image),
        _NORM_RECT_STREAM_NAME: packet_creator.create_proto(
            normalized_rect.to_pb2()
        ),
    })
    if output_packets[_STYLIZED_IMAGE_NAME].is_empty():
      return None
    return packet_getter.get_image(output_packets[_STYLIZED_IMAGE_NAME])
