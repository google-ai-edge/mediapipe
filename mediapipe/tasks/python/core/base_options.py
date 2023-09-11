# Copyright 2022 The MediaPipe Authors.
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
"""Base options for MediaPipe Task APIs."""

import dataclasses
import enum
import os
import platform
from typing import Any, Optional

from mediapipe.calculators.tensor import inference_calculator_pb2
from mediapipe.tasks.cc.core.proto import acceleration_pb2
from mediapipe.tasks.cc.core.proto import base_options_pb2
from mediapipe.tasks.cc.core.proto import external_file_pb2
from mediapipe.tasks.python.core.optional_dependencies import doc_controls

_DelegateProto = inference_calculator_pb2.InferenceCalculatorOptions.Delegate
_AccelerationProto = acceleration_pb2.Acceleration
_BaseOptionsProto = base_options_pb2.BaseOptions
_ExternalFileProto = external_file_pb2.ExternalFile


@dataclasses.dataclass
class BaseOptions:
  """Base options for MediaPipe Tasks' Python APIs.

  Represents external model asset used by the Task APIs. The files can be
  specified by one of the following two ways:

  (1) model asset file path in `model_asset_path`.
  (2) model asset contents loaded in `model_asset_buffer`.

  If more than one field of these fields is provided, they are used in this
  precedence order.

  Attributes:
    model_asset_path: Path to the model asset file.
    model_asset_buffer: The model asset file contents as bytes.
    delegate: Acceleration to use. Supported values are GPU and CPU. GPU support
      is currently limited to Ubuntu platforms.
  """

  class Delegate(enum.Enum):
    CPU = 0
    GPU = 1

  model_asset_path: Optional[str] = None
  model_asset_buffer: Optional[bytes] = None
  delegate: Optional[Delegate] = None

  @doc_controls.do_not_generate_docs
  def to_pb2(self) -> _BaseOptionsProto:
    """Generates a BaseOptions protobuf object."""
    if self.model_asset_path is not None:
      full_path = os.path.abspath(self.model_asset_path)
    else:
      full_path = None

    platform_name = platform.system()

    if self.delegate == BaseOptions.Delegate.GPU:
      if platform_name == 'Linux':
        acceleration_proto = _AccelerationProto(gpu=_DelegateProto.Gpu())
      else:
        raise NotImplementedError(
            'GPU Delegate is not yet supported for ' + platform_name
        )
    elif self.delegate == BaseOptions.Delegate.CPU:
      acceleration_proto = _AccelerationProto(tflite=_DelegateProto.TfLite())
    else:
      acceleration_proto = None

    return _BaseOptionsProto(
        model_asset=_ExternalFileProto(
            file_name=full_path, file_content=self.model_asset_buffer
        ),
        acceleration=acceleration_proto,
    )

  @classmethod
  @doc_controls.do_not_generate_docs
  def create_from_pb2(cls, pb2_obj: _BaseOptionsProto) -> 'BaseOptions':
    """Creates a `BaseOptions` object from the given protobuf object."""
    delegate = None
    if pb2_obj.acceleration is not None:
      delegate = (
          BaseOptions.Delegate.GPU
          if pb2_obj.acceleration.gpu is not None
          else BaseOptions.Delegate.CPU
      )

    return BaseOptions(
        model_asset_path=pb2_obj.model_asset.file_name,
        model_asset_buffer=pb2_obj.model_asset.file_content,
        delegate=delegate,
    )

  def __eq__(self, other: Any) -> bool:
    """Checks if this object is equal to the given object.

    Args:
      other: The object to be compared with.

    Returns:
      True if the objects are equal.
    """
    if not isinstance(other, BaseOptions):
      return False

    return self.to_pb2().__eq__(other.to_pb2())
