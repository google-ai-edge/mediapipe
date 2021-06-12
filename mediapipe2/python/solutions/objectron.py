# Copyright 2020-2021 The MediaPipe Authors.
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

"""MediaPipe Objectron."""

import enum
from typing import List, Tuple, NamedTuple, Optional

import attr
import numpy as np

from mediapipe.calculators.core import constant_side_packet_calculator_pb2
# pylint: disable=unused-import
from mediapipe.calculators.core import gate_calculator_pb2
from mediapipe.calculators.core import split_vector_calculator_pb2
from mediapipe.calculators.tensor import image_to_tensor_calculator_pb2
from mediapipe.calculators.tensor import inference_calculator_pb2
from mediapipe.calculators.tensor import tensors_to_detections_calculator_pb2
from mediapipe.calculators.tensor import tensors_to_floats_calculator_pb2
from mediapipe.calculators.tensor import tensors_to_landmarks_calculator_pb2
from mediapipe.calculators.tflite import ssd_anchors_calculator_pb2
from mediapipe.calculators.util import association_calculator_pb2
from mediapipe.calculators.util import collection_has_min_size_calculator_pb2
from mediapipe.calculators.util import detection_label_id_to_text_calculator_pb2
from mediapipe.calculators.util import detections_to_rects_calculator_pb2
from mediapipe.calculators.util import landmark_projection_calculator_pb2
from mediapipe.calculators.util import local_file_contents_calculator_pb2
from mediapipe.calculators.util import non_max_suppression_calculator_pb2
from mediapipe.calculators.util import rect_transformation_calculator_pb2
from mediapipe.calculators.util import thresholding_calculator_pb2
from mediapipe.framework.formats import landmark_pb2
from mediapipe.modules.objectron.calculators import annotation_data_pb2
from mediapipe.modules.objectron.calculators import frame_annotation_to_rect_calculator_pb2
from mediapipe.modules.objectron.calculators import lift_2d_frame_annotation_to_3d_calculator_pb2
# pylint: enable=unused-import
from mediapipe.python.solution_base import SolutionBase
from mediapipe.python.solutions import download_utils


class BoxLandmark(enum.IntEnum):
  """The 9 3D box landmarks."""
  #
  #       3 + + + + + + + + 7
  #       +\                +\          UP
  #       + \               + \
  #       +  \              +  \        |
  #       +   4 + + + + + + + + 8       | y
  #       +   +             +   +       |
  #       +   +             +   +       |
  #       +   +     (0)     +   +       .------- x
  #       +   +             +   +        \
  #       1 + + + + + + + + 5   +         \
  #        \  +              \  +          \ z
  #         \ +               \ +           \
  #          \+                \+
  #           2 + + + + + + + + 6
  CENTER = 0
  BACK_BOTTOM_LEFT = 1
  FRONT_BOTTOM_LEFT = 2
  BACK_TOP_LEFT = 3
  FRONT_TOP_LEFT = 4
  BACK_BOTTOM_RIGHT = 5
  FRONT_BOTTOM_RIGHT = 6
  BACK_TOP_RIGHT = 7
  FRONT_TOP_RIGHT = 8

BINARYPB_FILE_PATH = 'mediapipe/modules/objectron/objectron_cpu.binarypb'
BOX_CONNECTIONS = frozenset([
    (BoxLandmark.BACK_BOTTOM_LEFT, BoxLandmark.FRONT_BOTTOM_LEFT),
    (BoxLandmark.BACK_BOTTOM_LEFT, BoxLandmark.BACK_TOP_LEFT),
    (BoxLandmark.BACK_BOTTOM_LEFT, BoxLandmark.BACK_BOTTOM_RIGHT),
    (BoxLandmark.FRONT_BOTTOM_LEFT, BoxLandmark.FRONT_TOP_LEFT),
    (BoxLandmark.FRONT_BOTTOM_LEFT, BoxLandmark.FRONT_BOTTOM_RIGHT),
    (BoxLandmark.BACK_TOP_LEFT, BoxLandmark.FRONT_TOP_LEFT),
    (BoxLandmark.BACK_TOP_LEFT, BoxLandmark.BACK_TOP_RIGHT),
    (BoxLandmark.FRONT_TOP_LEFT, BoxLandmark.FRONT_TOP_RIGHT),
    (BoxLandmark.BACK_BOTTOM_RIGHT, BoxLandmark.FRONT_BOTTOM_RIGHT),
    (BoxLandmark.BACK_BOTTOM_RIGHT, BoxLandmark.BACK_TOP_RIGHT),
    (BoxLandmark.FRONT_BOTTOM_RIGHT, BoxLandmark.FRONT_TOP_RIGHT),
    (BoxLandmark.BACK_TOP_RIGHT, BoxLandmark.FRONT_TOP_RIGHT),
])


@attr.s(auto_attribs=True)
class ObjectronModel(object):
  model_path: str
  label_name: str


@attr.s(auto_attribs=True, frozen=True)
class ShoeModel(ObjectronModel):
  model_path: str = ('mediapipe/modules/objectron/'
                     'object_detection_3d_sneakers.tflite')
  label_name: str = 'Footwear'


@attr.s(auto_attribs=True, frozen=True)
class ChairModel(ObjectronModel):
  model_path: str = ('mediapipe/modules/objectron/'
                     'object_detection_3d_chair.tflite')
  label_name: str = 'Chair'


@attr.s(auto_attribs=True, frozen=True)
class CameraModel(ObjectronModel):
  model_path: str = ('mediapipe/modules/objectron/'
                     'object_detection_3d_camera.tflite')
  label_name: str = 'Camera'


@attr.s(auto_attribs=True, frozen=True)
class CupModel(ObjectronModel):
  model_path: str = ('mediapipe/modules/objectron/'
                     'object_detection_3d_cup.tflite')
  label_name: str = 'Coffee cup, Mug'

_MODEL_DICT = {
    'Shoe': ShoeModel(),
    'Chair': ChairModel(),
    'Cup': CupModel(),
    'Camera': CameraModel()
}


def _download_oss_objectron_models(objectron_model: str):
  """Downloads the objectron models from the MediaPipe Github repo if they don't exist in the package."""

  download_utils.download_oss_model(
      'mediapipe/modules/objectron/object_detection_ssd_mobilenetv2_oidv4_fp16.tflite'
  )
  download_utils.download_oss_model(objectron_model)


def get_model_by_name(name: str) -> ObjectronModel:
  if name not in _MODEL_DICT:
    raise ValueError(f'{name} is not a valid model name for Objectron.')
  _download_oss_objectron_models(_MODEL_DICT[name].model_path)
  return _MODEL_DICT[name]


@attr.s(auto_attribs=True)
class ObjectronOutputs(object):
  landmarks_2d: landmark_pb2.NormalizedLandmarkList
  landmarks_3d: landmark_pb2.LandmarkList
  rotation: np.ndarray
  translation: np.ndarray
  scale: np.ndarray


class Objectron(SolutionBase):
  """MediaPipe Objectron.

  MediaPipe Objectron processes an RGB image and returns the 3D box landmarks
  and 2D rectangular bounding box of each detected object.
  """

  def __init__(self,
               static_image_mode: bool = False,
               max_num_objects: int = 5,
               min_detection_confidence: float = 0.5,
               min_tracking_confidence: float = 0.99,
               model_name: str = 'Shoe',
               focal_length: Tuple[float, float] = (1.0, 1.0),
               principal_point: Tuple[float, float] = (0.0, 0.0),
               image_size: Optional[Tuple[int, int]] = None,
               ):
    """Initializes a MediaPipe Objectron class.

    Args:
      static_image_mode: Whether to treat the input images as a batch of static
        and possibly unrelated images, or a video stream.
      max_num_objects: Maximum number of objects to detect.
      min_detection_confidence: Minimum confidence value ([0.0, 1.0]) for object
        detection to be considered successful.
      min_tracking_confidence: Minimum confidence value ([0.0, 1.0]) for the
        box landmarks to be considered tracked successfully.
      model_name: Name of model to use for predicting box landmarks, currently
        support {'Shoe', 'Chair', 'Cup', 'Camera'}.
      focal_length: Camera focal length `(fx, fy)`, by default is defined in NDC
        space. To use focal length (fx_pixel, fy_pixel) in pixel space, users
        should provide image_size = (image_width, image_height) to enable
        conversions inside the API.
      principal_point: Camera principal point (px, py), by default is defined in
        NDC space. To use principal point (px_pixel, py_pixel) in pixel space,
        users should provide image_size = (image_width, image_height) to enable
        conversions inside the API.
      image_size (Optional): size (image_width, image_height) of the input image
        , ONLY needed when use focal_length and principal_point in pixel space.

    Raises:
      ConnectionError: If the objectron open source model can't be downloaded
        from the MediaPipe Github repo.
    """
    # Get Camera parameters.
    fx, fy = focal_length
    px, py = principal_point
    if image_size is not None:
      half_width = image_size[0] / 2.0
      half_height = image_size[1] / 2.0
      fx = fx / half_width
      fy = fy / half_height
      px = - (px - half_width) / half_width
      py = - (py - half_height) / half_height

    # Create and init model.
    model = get_model_by_name(model_name)
    super().__init__(
        binary_graph_path=BINARYPB_FILE_PATH,
        side_inputs={
            'box_landmark_model_path': model.model_path,
            'allowed_labels': model.label_name,
            'max_num_objects': max_num_objects,
        },
        calculator_params={
            'ConstantSidePacketCalculator.packet': [
                constant_side_packet_calculator_pb2
                .ConstantSidePacketCalculatorOptions.ConstantSidePacket(
                    bool_value=not static_image_mode)
            ],
            ('objectdetectionoidv4subgraph'
             '__TensorsToDetectionsCalculator.min_score_thresh'):
                min_detection_confidence,
            ('boxlandmarksubgraph__ThresholdingCalculator'
             '.threshold'):
                min_tracking_confidence,
            ('Lift2DFrameAnnotationTo3DCalculator'
             '.normalized_focal_x'): fx,
            ('Lift2DFrameAnnotationTo3DCalculator'
             '.normalized_focal_y'): fy,
            ('Lift2DFrameAnnotationTo3DCalculator'
             '.normalized_principal_point_x'): px,
            ('Lift2DFrameAnnotationTo3DCalculator'
             '.normalized_principal_point_y'): py,
        },
        outputs=['detected_objects'])

  def process(self, image: np.ndarray) -> NamedTuple:
    """Processes an RGB image and returns the box landmarks and rectangular bounding box of each detected object.

    Args:
      image: An RGB image represented as a numpy ndarray.

    Raises:
      RuntimeError: If the underlying graph throws any error.
      ValueError: If the input image is not three channel RGB.

    Returns:
      A NamedTuple object with a "detected_objects" field that contains a list
      of detected 3D bounding boxes. Each detected box is represented as an
      "ObjectronOutputs" instance.
    """

    results = super().process(input_data={'image': image})
    if results.detected_objects:
      results.detected_objects = self._convert_format(results.detected_objects)
    else:
      results.detected_objects = None
    return results

  def _convert_format(
      self,
      inputs: annotation_data_pb2.FrameAnnotation) -> List[ObjectronOutputs]:
    new_outputs = list()
    for annotation in inputs.annotations:
      # Get 3d object pose.
      rotation = np.reshape(np.array(annotation.rotation), (3, 3))
      translation = np.array(annotation.translation)
      scale = np.array(annotation.scale)
      # Get 2d/3d landmakrs.
      landmarks_2d = landmark_pb2.NormalizedLandmarkList()
      landmarks_3d = landmark_pb2.LandmarkList()
      for keypoint in annotation.keypoints:
        point_2d = keypoint.point_2d
        landmarks_2d.landmark.add(x=point_2d.x, y=point_2d.y)
        point_3d = keypoint.point_3d
        landmarks_3d.landmark.add(x=point_3d.x, y=point_3d.y, z=point_3d.z)

      # Add to objectron outputs.
      new_outputs.append(ObjectronOutputs(landmarks_2d, landmarks_3d,
                                          rotation, translation, scale=scale))
    return new_outputs
