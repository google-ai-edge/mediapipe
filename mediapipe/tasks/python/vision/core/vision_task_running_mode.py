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
