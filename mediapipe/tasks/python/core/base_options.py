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
"""Base options for MediaPipe Task APIs."""

import dataclasses
import os
from typing import Any, Optional

from mediapipe.tasks.cc.core.proto import base_options_pb2
from mediapipe.tasks.cc.core.proto import external_file_pb2
from mediapipe.tasks.python.core.optional_dependencies import doc_controls

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
  """

  model_asset_path: Optional[str] = None
  model_asset_buffer: Optional[bytes] = None
  # TODO: Allow Python API to specify acceleration settings.

  @doc_controls.do_not_generate_docs
  def to_pb2(self) -> _BaseOptionsProto:
    """Generates a BaseOptions protobuf object."""
    if self.model_asset_path is not None:
      full_path = os.path.abspath(self.model_asset_path)
    else:
      full_path = None

    return _BaseOptionsProto(
        model_asset=_ExternalFileProto(
            file_name=full_path, file_content=self.model_asset_buffer))

  @classmethod
  @doc_controls.do_not_generate_docs
  def create_from_pb2(cls, pb2_obj: _BaseOptionsProto) -> 'BaseOptions':
    """Creates a `BaseOptions` object from the given protobuf object."""
    return BaseOptions(
        model_asset_path=pb2_obj.model_asset.file_name,
        model_asset_buffer=pb2_obj.model_asset.file_content)

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
