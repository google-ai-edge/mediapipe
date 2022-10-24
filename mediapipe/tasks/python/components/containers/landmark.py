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
"""Landmark data class."""

import dataclasses
from typing import Any, Optional, List

from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks.python.core.optional_dependencies import doc_controls

_LandmarkProto = landmark_pb2.Landmark
_LandmarkListProto = landmark_pb2.LandmarkList
_NormalizedLandmarkProto = landmark_pb2.NormalizedLandmark
_NormalizedLandmarkListProto = landmark_pb2.NormalizedLandmarkList


@dataclasses.dataclass
class Landmark:
  """A landmark that can have 1 to 3 dimensions.

  Use x for 1D points, (x, y) for 2D points and (x, y, z) for 3D points.

  Attributes:
    x: The x coordinate of the 3D point.
    y: The y coordinate of the 3D point.
    z: The z coordinate of the 3D point.
    visibility: Landmark visibility. Should stay unset if not supported.
      Float score of whether landmark is visible or occluded by other objects.
      Landmark considered as invisible also if it is not present on the screen
      (out of scene bounds). Depending on the model, visibility value is either
      a sigmoid or an argument of sigmoid.
    presence: Landmark presence. Should stay unset if not supported.
      Float score of whether landmark is present on the scene (located within
      scene bounds). Depending on the model, presence value is either a result
      of sigmoid or an argument of sigmoid function to get landmark presence
      probability.
  """

  x: Optional[float] = None
  y: Optional[float] = None
  z: Optional[float] = None
  visibility: Optional[float] = None
  presence: Optional[float] = None

  @doc_controls.do_not_generate_docs
  def to_pb2(self) -> _LandmarkProto:
    """Generates a Landmark protobuf object."""
    return _LandmarkProto(
        x=self.x,
        y=self.y,
        z=self.z,
        visibility=self.visibility,
        presence=self.presence)

  @classmethod
  @doc_controls.do_not_generate_docs
  def create_from_pb2(cls, pb2_obj: _LandmarkProto) -> 'Landmark':
    """Creates a `Landmark` object from the given protobuf object."""
    return Landmark(
        x=pb2_obj.x,
        y=pb2_obj.y,
        z=pb2_obj.z,
        visibility=pb2_obj.visibility,
        presence=pb2_obj.presence)

  def __eq__(self, other: Any) -> bool:
    """Checks if this object is equal to the given object.

    Args:
      other: The object to be compared with.

    Returns:
      True if the objects are equal.
    """
    if not isinstance(other, Landmark):
      return False

    return self.to_pb2().__eq__(other.to_pb2())


@dataclasses.dataclass
class LandmarkList:
  """Represents the list of landmarks.

  Attributes:
    landmarks : A list of `Landmark` objects.
  """

  landmarks: List[Landmark]

  @doc_controls.do_not_generate_docs
  def to_pb2(self) -> _LandmarkListProto:
    """Generates a LandmarkList protobuf object."""
    return _LandmarkListProto(
        landmark=[
            landmark.to_pb2()
            for landmark in self.landmarks
        ]
    )

  @classmethod
  @doc_controls.do_not_generate_docs
  def create_from_pb2(
      cls,
      pb2_obj: _LandmarkListProto
  ) -> 'LandmarkList':
    """Creates a `LandmarkList` object from the given protobuf object."""
    return LandmarkList(
        landmarks=[
            Landmark.create_from_pb2(landmark)
            for landmark in pb2_obj.landmark
        ]
    )

  def __eq__(self, other: Any) -> bool:
    """Checks if this object is equal to the given object.
    Args:
      other: The object to be compared with.
    Returns:
      True if the objects are equal.
    """
    if not isinstance(other, LandmarkList):
      return False

    return self.to_pb2().__eq__(other.to_pb2())


@dataclasses.dataclass
class NormalizedLandmark:
  """A normalized version of above Landmark proto.

  All coordinates should be within [0, 1].

  Attributes:
    x: The normalized x coordinate of the 3D point.
    y: The normalized y coordinate of the 3D point.
    z: The normalized z coordinate of the 3D point.
    visibility: Landmark visibility. Should stay unset if not supported.
      Float score of whether landmark is visible or occluded by other objects.
      Landmark considered as invisible also if it is not present on the screen
      (out of scene bounds). Depending on the model, visibility value is either
      a sigmoid or an argument of sigmoid.
    presence: Landmark presence. Should stay unset if not supported.
      Float score of whether landmark is present on the scene (located within
      scene bounds). Depending on the model, presence value is either a result
      of sigmoid or an argument of sigmoid function to get landmark presence
      probability.
  """

  x: Optional[float] = None
  y: Optional[float] = None
  z: Optional[float] = None
  visibility: Optional[float] = None
  presence: Optional[float] = None

  @doc_controls.do_not_generate_docs
  def to_pb2(self) -> _NormalizedLandmarkProto:
    """Generates a NormalizedLandmark protobuf object."""
    return _NormalizedLandmarkProto(
      x=self.x,
      y=self.y,
      z=self.z,
      visibility=self.visibility,
      presence=self.presence)

  @classmethod
  @doc_controls.do_not_generate_docs
  def create_from_pb2(
      cls,
      pb2_obj: _NormalizedLandmarkProto
  ) -> 'NormalizedLandmark':
    """Creates a `NormalizedLandmark` object from the given protobuf object."""
    return NormalizedLandmark(
      x=pb2_obj.x,
      y=pb2_obj.y,
      z=pb2_obj.z,
      visibility=pb2_obj.visibility,
      presence=pb2_obj.presence)

  def __eq__(self, other: Any) -> bool:
    """Checks if this object is equal to the given object.

    Args:
      other: The object to be compared with.

    Returns:
      True if the objects are equal.
    """
    if not isinstance(other, NormalizedLandmark):
      return False

    return self.to_pb2().__eq__(other.to_pb2())


@dataclasses.dataclass
class NormalizedLandmarkList:
  """Represents the list of normalized landmarks.

  Attributes:
    landmarks : A list of `Landmark` objects.
  """

  landmarks: List[NormalizedLandmark]

  @doc_controls.do_not_generate_docs
  def to_pb2(self) -> _NormalizedLandmarkListProto:
    """Generates a NormalizedLandmarkList protobuf object."""
    return _NormalizedLandmarkListProto(
      landmark=[
        landmark.to_pb2()
        for landmark in self.landmarks
      ]
    )

  @classmethod
  @doc_controls.do_not_generate_docs
  def create_from_pb2(
      cls,
      pb2_obj: _NormalizedLandmarkListProto
  ) -> 'NormalizedLandmarkList':
    """Creates a `NormalizedLandmarkList` object from the given protobuf object."""
    return NormalizedLandmarkList(
      landmarks=[
        NormalizedLandmark.create_from_pb2(landmark)
        for landmark in pb2_obj.landmark
      ]
    )

  def __eq__(self, other: Any) -> bool:
    """Checks if this object is equal to the given object.
    Args:
      other: The object to be compared with.
    Returns:
      True if the objects are equal.
    """
    if not isinstance(other, NormalizedLandmarkList):
      return False

    return self.to_pb2().__eq__(other.to_pb2())
