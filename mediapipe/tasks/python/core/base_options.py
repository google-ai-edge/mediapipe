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
import platform
import sys
import types
from typing import Any, Optional

import certifi

from mediapipe.tasks.python.core import base_options_c as base_options_c_lib
from mediapipe.tasks.python.core.optional_dependencies import doc_controls



# C enum value for host environment.
HOST_ENVIRONMENT_PYTHON: int = 3


class _HostSystem(enum.IntEnum):
  """An mapping for the C enum values for the host operating system.

  Attributes:
    HOST_SYSTEM_UNKNOWN: Unknown host system.
    HOST_SYSTEM_LINUX: Linux host system.
    HOST_SYSTEM_MAC: MacOS host system.
    HOST_SYSTEM_WINDOWS: Windows host system.
  """

  HOST_SYSTEM_UNKNOWN = 0
  HOST_SYSTEM_LINUX = 1
  HOST_SYSTEM_MAC = 2
  HOST_SYSTEM_WINDOWS = 3




_HOST_SYSTEM_BY_PLATFORM = types.MappingProxyType({
    'Darwin': _HostSystem.HOST_SYSTEM_MAC,
    'Windows': _HostSystem.HOST_SYSTEM_WINDOWS,
    'Linux': _HostSystem.HOST_SYSTEM_LINUX,
})


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

  class Delegate(enum.IntEnum):
    CPU = 0
    GPU = 1

  model_asset_path: Optional[str] = None
  model_asset_buffer: Optional[bytes] = None
  delegate: Optional[Delegate] = None

  @doc_controls.do_not_generate_docs
  def to_ctypes(self) -> base_options_c_lib.BaseOptionsC:
    """Creates a BaseOptionsC struct from the BaseOptions object."""
    ca_bundle_path = certifi.where()
    host_system = _HOST_SYSTEM_BY_PLATFORM.get(
        platform.system(), _HostSystem.HOST_SYSTEM_UNKNOWN
    )
    options = base_options_c_lib.BaseOptionsC()
    options.model_asset_buffer = self.model_asset_buffer
    options.model_asset_buffer_count = (
        len(self.model_asset_buffer) if self.model_asset_buffer else 0
    )
    options.model_asset_path = (
        self.model_asset_path.encode('utf-8') if self.model_asset_path else None
    )
    options.delegate = self.delegate.value if self.delegate else 0
    options.host_environment = HOST_ENVIRONMENT_PYTHON
    host_version, *_ = sys.version.split(' ')
    options.host_system = host_system
    options.host_version = host_version.encode('utf-8')
    options.ca_bundle_path = ca_bundle_path.encode('utf-8')
    return options

  def __eq__(self, other: Any) -> bool:
    """Checks if this object is equal to the given object.

    Args:
      other: The object to be compared with.

    Returns:
      True if the objects are equal.
    """
    if not isinstance(other, BaseOptions):
      return False
    return (
        self.model_asset_path == other.model_asset_path
        and self.model_asset_buffer == other.model_asset_buffer
        and self.delegate == other.delegate
    )
