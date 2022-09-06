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
from typing import Any, Optional

from mediapipe.tasks.cc.core.proto import base_options_pb2
from mediapipe.tasks.cc.core.proto import external_file_pb2
from mediapipe.tasks.python.core.optional_dependencies import doc_controls

_BaseOptionsProto = base_options_pb2.BaseOptions
_ExternalFileProto = external_file_pb2.ExternalFile


@dataclasses.dataclass
class BaseOptions:
  """Base options for MediaPipe Tasks' Python APIs.

  Represents external files used by the Task APIs (e.g. TF Lite FlatBuffer or
  plain-text labels file). The files can be specified by one of the following
  two ways:

  (1) file contents loaded in `file_content`.
  (2) file path in `file_name`.

  If more than one field of these fields is provided, they are used in this
  precedence order.

  Attributes:
    file_name: Path to the index.
    file_content: The index file contents as bytes.
  """

  file_name: Optional[str] = None
  file_content: Optional[bytes] = None
  # TODO: Allow Python API to specify acceleration settings.

  @doc_controls.do_not_generate_docs
  def to_pb2(self) -> _BaseOptionsProto:
    """Generates a BaseOptions protobuf object."""
    return _BaseOptionsProto(
        model_file=_ExternalFileProto(
            file_name=self.file_name, file_content=self.file_content))

  @classmethod
  @doc_controls.do_not_generate_docs
  def create_from_pb2(cls, pb2_obj: _BaseOptionsProto) -> 'BaseOptions':
    """Creates a `BaseOptions` object from the given protobuf object."""
    return BaseOptions(
        file_name=pb2_obj.model_file.file_name,
        file_content=pb2_obj.model_file.file_content)

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
