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
"""The running mode of MediaPipe Vision Tasks."""

import enum
import types
from typing import Any, Callable, Mapping, Optional

from mediapipe.tasks.python.core.optional_dependencies import doc_controls

# The C API constants that map to the Python enum values.
_CTYPE_VALUE_MAP = types.MappingProxyType({
    'IMAGE': 1,
    'VIDEO': 2,
    'LIVE_STREAM': 3,
})


class VisionTaskRunningMode(enum.Enum):
  """MediaPipe vision task running mode.

  Attributes:
    IMAGE: The mode for running a mediapipe vision task on single image inputs.
    VIDEO: The mode for running a mediapipe vision task on the decoded frames
      of an input video.
    LIVE_STREAM: The mode for running a mediapipe vision task on a live stream
      of input data, such as from camera.
  """
  IMAGE = 'IMAGE'
  VIDEO = 'VIDEO'
  LIVE_STREAM = 'LIVE_STREAM'

  @property
  @doc_controls.do_not_generate_docs
  def ctype(self) -> int:
    """Generates a C API int object."""
    ctype_value = _CTYPE_VALUE_MAP.get(self.value)
    assert (
        ctype_value is not None
    ), f'Unsupported vision task running mode: {self}'
    return ctype_value


def validate_running_mode(
    running_mode: VisionTaskRunningMode,
    packet_callback: Optional[Callable[[Mapping[str, Any]], None]] = None,
) -> None:
  """Validates the running mode of the mediapipe vision task.

  Args:
    running_mode: The running mode of the mediapipe vision task.
    packet_callback: The user-defined result callback.

  Raises:
    ValueError: If packet callback is provided in image or video mode or
    missing in live stream mode.
  """
  if running_mode == VisionTaskRunningMode.LIVE_STREAM:
    if packet_callback is None:
      raise ValueError(
          'The vision task is in live stream mode, a user-defined result '
          'callback must be provided.'
      )
  elif packet_callback:
    raise ValueError(
        'The vision task is in image or video mode, a user-defined result '
        'callback should not be provided.'
    )
